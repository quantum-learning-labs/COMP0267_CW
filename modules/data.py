"""SVO-Probes: download, parse and embed once; then load, derive the swap set and serve batches.

Each row pairs a caption (``corrected_sentence``) with a positive image that it describes and a negative image that
differs in the subject, the verb or the object (``subj_neg``, ``verb_neg``, ``obj_neg``). Images are represented by
their CLIP ViT-B/32 embeddings, keyed by image id.
"""
import io
import re
import time
import zipfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen, urlretrieve

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from modules.diagrams import parse

CSV_URL = "https://raw.githubusercontent.com/google-deepmind/svo_probes/main/svo_probes.csv"
MANIFEST = Path(__file__).resolve().parent / "manifests" / "svo-probes" / "curated.csv.gz"

# verbs whose subject and object can be swapped without changing the meaning: no negatives for the swap set
SYMMETRIC_VERBS = {"meet", "hug", "kiss", "marry", "touch", "fight", "play", "talk", "dance", "race", "wrestle",
                   "greet", "shake", "face", "resemble", "join", "accompany", "chat", "argue", "embrace", "cuddle",
                   "date"}


def download_svo_probes(raw_dir: Path) -> pd.DataFrame:
    """The curated SVO-Probes rows (with their ``legacy_split``) and subject/verb/object columns."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "svo_probes.csv"
    if not csv_path.exists():
        urlretrieve(CSV_URL, csv_path)
    manifest = pd.read_csv(MANIFEST)
    df = pd.read_csv(csv_path).iloc[manifest["csv_row"]].reset_index(drop=True)
    for col in manifest.columns.drop("csv_row"):
        df[col] = manifest[col].to_numpy()
    df[["subj", "verb", "obj"]] = df["pos_triplet"].str.split(",", expand=True)
    return df


def _fetch(url: str, timeout: int = 30, tries: int = 3) -> bytes | None:
    for attempt in range(tries):
        try:
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=timeout) as response:
                image = Image.open(io.BytesIO(response.read())).convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=92)
            return buffer.getvalue()
        except HTTPError as err:
            if err.code in (403, 404, 410):
                return None
        except Exception:
            pass
        time.sleep(1 + attempt)
    return None


def download_images(df: pd.DataFrame, zip_path: Path, workers: int = 32) -> None:
    """Fetch every positive and negative image into ``zip_path`` as ``images/<id>.jpg`` (dead links are skipped)."""
    urls = pd.concat([df[["pos_image_id", "pos_url"]].set_axis(["id", "url"], axis=1),
                      df[["neg_image_id", "neg_url"]].set_axis(["id", "url"], axis=1)]).drop_duplicates("id")
    with zipfile.ZipFile(zip_path, "a") as archive:
        stored = set(archive.namelist())
        todo = urls[~urls["id"].map(lambda i: f"images/{i}.jpg" in stored)]
        with ThreadPoolExecutor(workers) as pool:
            for image_id, data in tqdm(zip(todo["id"], pool.map(_fetch, todo["url"])), total=len(todo), desc="images"):
                if data is not None:
                    archive.writestr(f"images/{image_id}.jpg", data)


@torch.no_grad()
def embed_images(zip_path: Path, device: str = "cpu") -> dict[int, torch.Tensor]:
    """CLIP ViT-B/32 embedding of every image in the archive, keyed by image id."""
    import clip  # optional dependency: only needed to rebuild the embeddings

    model, preprocess = clip.load("ViT-B/32", device=device)
    embeddings = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in tqdm([n for n in archive.namelist() if n.endswith(".jpg")], desc="CLIP"):
            image = preprocess(Image.open(io.BytesIO(archive.read(name))).convert("RGB")).unsqueeze(0).to(device)
            embeddings[int(Path(name).stem)] = model.encode_image(image).squeeze(0).float().cpu()
    return embeddings


def prepare_svo_probes(root: Path, device: str = "cpu") -> None:
    """Build ``processed/all.pkl`` (rows + parsed diagrams) and ``processed/image_embeddings.pt`` from scratch."""
    raw, processed = root / "svo-probes" / "raw", root / "svo-probes" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    df = download_svo_probes(raw)
    download_images(df, raw / "images.zip")
    torch.save(embed_images(raw / "images.zip", device), processed / "image_embeddings.pt")
    df["corrected_sentence_diagram"] = parse(df["corrected_sentence"].tolist())
    df.to_pickle(processed / "all.pkl")


def load_svo_probes(root: Path) -> tuple[pd.DataFrame, dict[int, torch.Tensor]]:
    """Rows with a diagram and both image embeddings, and the embeddings; prepares the data first if missing."""
    processed = root / "svo-probes" / "processed"
    if not (processed / "all.pkl").exists() or not (processed / "image_embeddings.pt").exists():
        prepare_svo_probes(root)
    df = pd.read_pickle(processed / "all.pkl")
    embeddings = torch.load(processed / "image_embeddings.pt")
    keep = (df["corrected_sentence_diagram"].notna() & df["pos_image_id"].isin(embeddings)
            & df["neg_image_id"].isin(embeddings))
    return df[keep].reset_index(drop=True), embeddings


def build_swap_set(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose caption has exactly two distinct nouns and a non-symmetric verb, with the nouns swapped.

    The swapped caption (``swapped_sentence``, ``swapped_sentence_diagram``) uses the same words and grammar as the
    original, so only the way the words are composed tells the two apart.
    """
    rows = []
    for _, row in df.iterrows():
        diagram = row["corrected_sentence_diagram"]
        nouns = [i for i, (_, _, types) in enumerate(diagram) if list(types) == ["n"]]
        if len(nouns) != 2 or row["verb"].lower() in SYMMETRIC_VERBS:
            continue
        (a, b), (word_a, word_b) = nouns, (diagram[nouns[0]][0], diagram[nouns[1]][0])
        if word_a == word_b:
            continue
        swapped = [(word, list(wires), list(types)) for word, wires, types in diagram]
        swapped[a], swapped[b] = (word_b, *swapped[a][1:]), (word_a, *swapped[b][1:])
        sentence = re.sub(rf"\b{word_a}\b", "\0", row["corrected_sentence"].lower(), count=1)
        sentence = re.sub(rf"\b{word_b}\b", word_a, sentence, count=1).replace("\0", word_b)
        rows.append({**row.to_dict(), "swapped_sentence": sentence, "swapped_sentence_diagram": swapped})
    return pd.DataFrame(rows).reset_index(drop=True)


class ChoiceDataset(Dataset):
    """(caption, image) pairs plus one negative: an image (``negative_kind='image'``) or a caption (``'caption'``).

    Captions are compiled recipes; images are looked up in ``embeddings`` by id.
    """

    def __init__(self, captions: Sequence, images: Sequence[int], embeddings: dict[int, torch.Tensor],
                 negatives: Sequence, negative_kind: str = "image") -> None:
        if negative_kind not in ("image", "caption"):
            raise ValueError("negative_kind must be 'image' or 'caption'")
        self.captions, self.images, self.negatives = list(captions), list(images), list(negatives)
        self.embeddings, self.negative_kind = embeddings, negative_kind

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, i: int) -> dict:
        negative = self.negatives[i]
        return {"caption": self.captions[i], "image": self.embeddings[int(self.images[i])],
                f"negative_{self.negative_kind}":
                    self.embeddings[int(negative)] if self.negative_kind == "image" else negative}


def collate(batch: list[dict]) -> dict:
    """Stack tensors, keep everything else (recipes) as lists."""
    return {key: torch.stack(values) if isinstance(values[0], torch.Tensor) else values
            for key, values in ((key, [item[key] for item in batch]) for key in batch[0])}
