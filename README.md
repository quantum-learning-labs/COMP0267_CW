# COMP0267 Coursework 2026-2027

Compositional text–image models on SVO-Probes: sentences are parsed into string diagrams, compiled into tensor
networks or parametrised quantum circuits, trained with InfoNCE against frozen CLIP image features, and evaluated by
exact simulation, by noiseless and noisy emulation, and optionally on quantum hardware.

The coursework is `notebooks/tutorial.ipynb`, which states the questions and marks.

## Quick start on Google Colab
**1. Get the notebook into Colab.**
- Open it directly: <https://colab.research.google.com/github/quantum-learning-labs/COMP0267_CW/blob/main/notebooks/tutorial.ipynb>,
  or in Colab choose *File → Open notebook → GitHub*, enter `quantum-learning-labs/COMP0267_CW` and pick
  `notebooks/tutorial.ipynb`.
- Choose *File → Save a copy in Drive* straight away and work in that copy. Otherwise your answers are lost when
  the session ends.
- The notebook runs on both CPU and GPU runtimes (*Runtime → Change runtime type*).

**2. Get the `comp0267` package.** Run the first code cell (*0. Setup*):
```python
%pip install -q "git+https://github.com/quantum-learning-labs/COMP0267_CW@main"
from comp0267.utils import setup

setup()
```
It installs the package from this repository with its requirements (about two minutes), and `setup()` prepares the
runtime (step 4). Run it again whenever Colab starts a new session (after a disconnect or a long idle). Never edit the
package. Build on it in the notebook by subclassing its classes:
```python
from comp0267.compilation.ansatz import Ansatz

class MyAnsatz(Ansatz):
    def word_circuit(self, circuit, wires, name):
        ...
```

**3. Get and set up Bobcat.** There is nothing to install. lambeq's own model server is offline, so do not call
`lambeq.BobcatParser()` yourself. Use
```python
from comp0267.symbolic.diagrams import load_parser, parse

parser = load_parser()           # first call per session downloads the model (0.8 GB, about a minute)
parse(["A dog chases a ball."], parser)
```
The model is cached in `~/.cache/comp0267` and reused for the rest of the session.
The processed SVO-Probes data (15 MB) is downloaded the same way by `comp0267.data.svo_probes.load_svo_probes()`, and
`comp0267.data.svo_probes.svo_swap()` derives the SVO-Swap set from it.

**4. Get and set up the packages.** Step 2 installs everything in `requirements.txt`:
- Colab's own torch, numpy, pandas and matplotlib are kept.
- It adds lambeq 0.5, Qiskit 2.1, Qiskit Aer 0.17, Qiskit IBM Runtime 0.45 and IQM's `iqm-client[qiskit]`
  (IQM's Qiskit adapter needs Qiskit below 2.2, which sets the other versions).
- It downgrades transformers to 4.x, which Bobcat needs.
- On a GPU runtime, `setup()` replaces Qiskit Aer by its GPU build of the same version (the CUDA 11 build, since the
  CUDA 12 build starts at Aer 0.18), so noiseless emulation runs on the GPU.

No restart is needed.

| problem | fix |
|---|---|
| `ModuleNotFoundError: No module named 'comp0267'` | new session: run the setup cell again |
| a download stops half-way | run the cell again (incomplete downloads are discarded) |
| `RuntimeError: could not download ...` | the course assets are not published yet (see *Course assets* below) |

## Local setup
Python 3.11–3.13:
```bash
python -m venv .venv && source .venv/bin/activate
pip install .            # or: pip install -e . to develop the package itself
```
The processed data (15 MB) and the Bobcat parser model (0.8 GB) are downloaded on first use and cached in
`~/.cache/comp0267` (`COMP0267_CACHE` moves the cache, `COMP0267_ASSETS` points to a mirror). lambeq's own model server
no longer exists, so `comp0267.symbolic.diagrams.load_parser()` is the way to get a working `BobcatParser`.

Students build on the package by subclassing its base classes in the notebook (e.g. `comp0267.compilation.ansatz.Ansatz`,
`comp0267.compilation.einsum.EinsumModel`), not by editing it.

## Course assets (instructors)
`comp0267.data.assets` downloads two archives from the GitHub release `assets-v1` of this repository:
`svo-probes-processed.tar.gz` (rows, parsed diagrams and CLIP image embeddings) and `bobcat.tar` (the Bobcat model in
fp16 safetensors). To publish them, create a release with tag `assets-v1` (*Releases → Draft a new release*) and attach
both files. `COMP0267_ASSETS` overrides the location.
