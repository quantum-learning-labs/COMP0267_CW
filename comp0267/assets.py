"""Course assets (processed SVO-Probes data, Bobcat parser model): downloaded once, then read from a local cache.

Set ``COMP0267_ASSETS`` to use a mirror (any URL urllib can read, including ``file://``) and ``COMP0267_CACHE`` to move
the cache.
"""
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

from tqdm.auto import tqdm

ASSET_URL = os.environ.get("COMP0267_ASSETS", "https://huggingface.co/datasets/quantum-learning-labs/comp0267/resolve/main")
CACHE = Path(os.environ.get("COMP0267_CACHE", Path.home() / ".cache" / "comp0267"))
ARCHIVES = {"svo-probes": "svo-probes-processed.tar.gz", "bobcat": "bobcat.tar"}


def fetch(name: str) -> Path:
    """Directory holding asset ``name`` (a key of ``ARCHIVES``), downloading and unpacking it on first use."""
    target = CACHE / name
    if (target / ".complete").exists():
        return target
    url = f"{ASSET_URL.rstrip('/')}/{ARCHIVES[name]}"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        archive = Path(tmp) / ARCHIVES[name]
        with urlopen(url) as response, open(archive, "wb") as out, \
                tqdm(total=int(response.headers.get("Content-Length") or 0) or None, unit="B", unit_scale=True,
                     desc=f"downloading {name}") as bar:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                bar.update(len(chunk))
        with tarfile.open(archive) as tar:
            tar.extractall(Path(tmp) / name, filter="data")
        shutil.rmtree(target, ignore_errors=True)
        (Path(tmp) / name).rename(target)
    (target / ".complete").touch()
    return target
