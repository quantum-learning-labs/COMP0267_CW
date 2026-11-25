"""Course assets (processed SVO-Probes data, Bobcat parser model): downloaded once, then read from a local cache.

Set ``COMP0267_ASSETS`` to use a mirror (any URL urllib can read, including ``file://``) and ``COMP0267_CACHE`` to move
the cache.
"""
import os, shutil, tarfile, tempfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from tqdm.auto import tqdm

ASSET_URL = os.environ.get("COMP0267_ASSETS",
                           "https://github.com/quantum-learning-labs/COMP0267_CW/releases/download/assets-v1")
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
        try:
            response = urlopen(url)
        except HTTPError as err:
            raise RuntimeError(f"could not download {name} from {url} (HTTP {err.code}); set COMP0267_ASSETS to a "
                               "mirror that holds it") from None
        with response, open(archive, "wb") as out, \
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
