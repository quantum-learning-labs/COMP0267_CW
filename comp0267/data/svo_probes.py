"""SVO-Probes: load the processed data, rebuild the SVO-Swap set and serve batches.

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

from comp0267.data.assets import fetch

# SVO-Swap (Lo et al., 2025, "DisCoCLIP"): 95 caption pairs from the val and test rows, with subject and object swapped
# where both are people or animals. Its generator was not released; this rule rebuilds the same 95 rows: two different
# people/animal nouns that are the image triplet's subject and object in that order, without the five person-woman
# captions (the two nouns can be one person), plus one caption the set keeps although its triplet is reversed.
PEOPLE_ANIMALS = {"man", "woman", "girl", "boy", "person", "child", "kid", "baby", "mother", "father", "parent", "lady",
                  "bride", "businessman", "player", "friend", "model", "couple", "dog", "puppy", "cat", "animal"}
SWAP_DROP_NOUNS = {("person", "woman")}
SWAP_KEEP = {("The person drives the man.", "man,drive,person")}


def load_svo_probes() -> tuple[pd.DataFrame, dict[int, torch.Tensor]]:
    """Rows with a parsed diagram and both image embeddings (``split`` is train, val or test), and the embeddings.

    The processed data is downloaded once and cached (see ``comp0267.data.assets``).
    """
    processed = fetch("svo-probes")
    df = pd.read_pickle(processed / "all.pkl").rename(columns={"legacy_split": "split"})
    embeddings = torch.load(processed / "image_embeddings.pt")
    keep = (df["corrected_sentence_diagram"].notna() & df["pos_image_id"].isin(embeddings)
            & df["neg_image_id"].isin(embeddings))
    return df[keep].reset_index(drop=True), embeddings


def _nouns(row: pd.Series) -> list[str]:
    """The caption's noun boxes (type n), in sentence order."""
    words = re.findall(r"[a-z']+", row["corrected_sentence"].lower())
    nouns = [word.lower() for word, _, types in row["corrected_sentence_diagram"] if list(types) == ["n"]]
    return sorted(nouns, key=lambda w: words.index(w) if w in words else len(words))


def svo_swap(df: pd.DataFrame) -> pd.DataFrame:
    """SVO-Swap: its 95 rows of ``df`` (from ``load_svo_probes``), each with the caption's two nouns swapped.

    The swapped caption (``swapped_sentence``, ``swapped_sentence_diagram``) uses the same words and grammar as the
    original, so only the way the words are composed tells the two apart.
    """
    rows = []
    for _, row in df[df["split"].isin(["val", "test"])].iterrows():
        nouns = _nouns(row)
        rule = (len(nouns) == 2 and nouns[0] != nouns[1] and all(n in PEOPLE_ANIMALS for n in nouns)
                and nouns == [str(row["subj"]).lower(), str(row["obj"]).lower()] and tuple(nouns) not in SWAP_DROP_NOUNS)
        if not (rule or (row["corrected_sentence"], row["pos_triplet"]) in SWAP_KEEP):
            continue
        diagram = row["corrected_sentence_diagram"]
        a, b = [i for i, (_, _, types) in enumerate(diagram) if list(types) == ["n"]]
        word_a, word_b = diagram[a][0], diagram[b][0]
        swapped = [(word, list(wires), list(types)) for word, wires, types in diagram]
        swapped[a], swapped[b] = (word_b, *swapped[a][1:]), (word_a, *swapped[b][1:])
        sentence = re.sub(rf"\b{re.escape(word_a)}\b", "\0", row["corrected_sentence"], count=1)
        sentence = re.sub(rf"\b{re.escape(word_b)}\b", word_a, sentence, count=1).replace("\0", word_b)
        rows.append({**row.to_dict(), "swapped_sentence": sentence, "swapped_sentence_diagram": swapped})
    if len(rows) != 95:
        raise ValueError(f"SVO-Swap rule found {len(rows)} rows, not 95: the data differs from the course's SVO-Probes")
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
