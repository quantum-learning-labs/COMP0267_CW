# COMP0267 Coursework 2026-2027

Compositional text–image models on SVO-Probes: sentences are parsed into string diagrams, compiled into tensor
networks or parametrised quantum circuits, trained with InfoNCE against frozen CLIP image features, and evaluated by
exact simulation, by noiseless and noisy emulation, and optionally on quantum hardware.

The coursework is `notebooks/tutorial.ipynb`; it states the questions and marks.

## Quick start on Google Colab
**1. Get the notebook into Colab.**
- Open it directly: <https://colab.research.google.com/github/quantum-learning-labs/COMP0267_CW/blob/main/notebooks/tutorial.ipynb>,
  or in Colab choose *File → Open notebook → GitHub*, enter `quantum-learning-labs/COMP0267_CW` and pick
  `notebooks/tutorial.ipynb`.
- Choose *File → Save a copy in Drive* straight away and work in that copy. Otherwise your answers are lost when
  the session ends.
- Choose *Runtime → Change runtime type → T4 GPU*. The quantum emulators (Part 4) and the parser run faster on it.
  A CPU runtime also works.

**2. Get the `comp0267` package.** Run the first code cell (*0. Setup*). It installs the package from this repository
with `pip`, which takes about two minutes. Run it again whenever Colab starts a new session (after a disconnect or a
long idle); it does nothing if the package is already installed. Never edit the package; build on it in the notebook
by subclassing its classes:
```python
from comp0267.quantum import Ansatz

class MyAnsatz(Ansatz):
    def word_circuit(self, circuit, wires, name):
        ...
```

**3. Get and set up Bobcat.** There is nothing to install. lambeq's own model server is offline, so do not call
`lambeq.BobcatParser()` yourself; use
```python
from comp0267.diagrams import load_parser, parse

parser = load_parser()           # first call per session downloads the model (0.8 GB, about a minute)
parse(["A dog chases a ball."], parser)
```
The model is cached in `~/.cache/comp0267` and reused for the rest of the session. It runs on the GPU if there is one.
The processed SVO-Probes data (15 MB) is downloaded the same way by `comp0267.data.load_svo_probes()`.

**4. Get and set up the packages.** Step 2 installs everything in `requirements.txt`:
- Colab's own torch, numpy, pandas and matplotlib are kept;
- it adds lambeq 0.5, Qiskit 2.4, Qiskit Aer 0.18 and Qiskit IBM Runtime 0.47;
- it downgrades transformers to 4.x, which Bobcat needs;
- on a GPU runtime it swaps Qiskit Aer for its GPU build of the same version.

No restart is needed. The second code cell prints `training on cpu; GPU available: True` on a GPU runtime (training
uses the CPU on purpose: these small models train about twice as fast there), and Part 4 prints `emulating on GPU`.

| problem | fix |
|---|---|
| `ModuleNotFoundError: No module named 'comp0267'` | new session: run the setup cell again |
| Part 4 prints `emulating on CPU` on a GPU runtime | *Runtime → Restart session*, then run from the top |
| a download stops half-way | run the cell again; incomplete downloads are discarded |

## Local setup
Python 3.11–3.13:
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
