"""SVO-Probes: load the processed data, derive the SVO-Swap set and serve batches.

Each row pairs a caption (``corrected_sentence``) with a positive image that it describes and a negative image that
differs in the subject, the verb or the object (``subj_neg``, ``verb_neg``, ``obj_neg``). Images are represented by
their CLIP ViT-B/32 embeddings, keyed by image id.
"""
import io, re
from collections.abc import Sequence
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset

from comp0267.assets import fetch

# verbs whose subject and object can be swapped without changing the meaning: left out of SVO-Swap
SYMMETRIC_VERBS = {"meet", "hug", "kiss", "marry", "touch", "fight", "play", "talk", "dance", "race", "wrestle",
                   "greet", "shake", "face", "resemble", "join", "accompany", "chat", "argue", "embrace", "cuddle",
                   "date"}


def load_svo_probes() -> tuple[pd.DataFrame, dict[int, torch.Tensor]]:
    """Rows with a parsed diagram and both image embeddings (``split`` is train, val or test), and the embeddings.

    The processed data is downloaded once and cached (see ``comp0267.assets``).
    """
    processed = fetch("svo-probes")
    df = pd.read_pickle(processed / "all.pkl").rename(columns={"legacy_split": "split"})
    embeddings = torch.load(processed / "image_embeddings.pt")
    keep = (df["corrected_sentence_diagram"].notna() & df["pos_image_id"].isin(embeddings)
            & df["neg_image_id"].isin(embeddings))
    return df[keep].reset_index(drop=True), embeddings


def svo_swap(df: pd.DataFrame) -> pd.DataFrame:
    """SVO-Swap: rows of ``df`` whose caption has two distinct nouns and a non-symmetric verb, with the nouns swapped.

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


def _image(url: str) -> Image.Image | None:
    try:
        with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=10) as response:
            return Image.open(io.BytesIO(response.read())).convert("RGB")
    except Exception:
        return None


def show_example(row: pd.Series) -> None:
    """Plot a row's positive and negative image under its caption (images are fetched from their original URLs)."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, side, colour in zip(axes, ("pos", "neg"), ("tab:green", "tab:red")):
        image = _image(row[f"{side}_url"])
        if image is None:
            ax.text(0.5, 0.5, "image no longer online", ha="center", va="center")
        else:
            ax.imshow(image)
        ax.set_xticks([]), ax.set_yticks([])
        ax.spines[:].set(color=colour, linewidth=4)
        triplet = row[f"{side}_triplet"].replace(",", ", ")
        ax.set_title(f"{'positive' if side == 'pos' else 'negative'}: ({triplet})", color=colour)
    fig.suptitle(f"“{row['corrected_sentence']}”")
    fig.tight_layout()
    plt.show()


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
