# COMP0267 Coursework 2026-2027

Compositional text–image models on SVO-Probes: sentences are parsed into string diagrams, compiled into tensor
networks or parametrised quantum circuits, trained with InfoNCE against frozen CLIP image features, and evaluated by
exact simulation, by noiseless and noisy emulation, and optionally on quantum hardware.

The coursework is `notebooks/tutorial.ipynb`; it states the questions and marks.

## Setup
**Colab:** open the notebook and run it; its first cell installs the `comp0267` package and its requirements.

**Locally** (Python 3.11–3.13):
```bash
python -m venv .venv && source .venv/bin/activate
pip install .            # or: pip install -e ".[test]" to develop the package itself
python -m pytest tests
```
The processed data (30 MB) and the Bobcat parser model (0.8 GB) are downloaded on first use and cached in
`~/.cache/comp0267` (`COMP0267_CACHE` moves the cache, `COMP0267_ASSETS` points to a mirror). lambeq's own model server
no longer exists, so `comp0267.diagrams.load_parser()` is the way to get a working `BobcatParser`.

Students build on the package by subclassing its base classes in the notebook (e.g. `comp0267.quantum.Ansatz`,
`comp0267.einsum.EinsumModel`), not by editing it.

## Layout
| file | contents |
|---|---|
| `comp0267/diagrams.py` | CCG parse trees -> string diagrams `(word, wires, types)`; the shared `ROOT` readout box; `load_parser` |
| `comp0267/einsum.py` | `Recipe` (einsum string + `Op` per tensor) and `EinsumModel`, which contracts batches of recipes |
| `comp0267/classical.py` | `TNCompiler` (diagram -> tensor network, optional MPS), `TNModel`, `PCA` image projection |
| `comp0267/quantum.py` | gates, `CircuitBuilder`, `Ansatz` (diagram -> circuit), `VQCModel`, `AngleEncoder` image map |
| `comp0267/circuits.py` | recipes -> Qiskit circuits, compute–uncompute overlap circuits, exact and sampled overlaps |
| `comp0267/training.py` | cosine and fidelity similarities, `InfoNCE`, `choice_scores` |
| `comp0267/data.py` | SVO-Probes load (and rebuild from the sources), swap set, `ChoiceDataset` |
| `comp0267/assets.py` | download-once cache for the data and the parser model |
| `tests/` | checks that the torch contractions agree with Qiskit statevectors |
