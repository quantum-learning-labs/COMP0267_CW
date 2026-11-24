# COMP0267 Coursework 2026-2027

Compositional text–image models on SVO-Probes: sentences are parsed into string diagrams, compiled into tensor
networks or parametrised quantum circuits, trained with InfoNCE against frozen CLIP image features, and evaluated by
exact simulation, by noiseless and noisy emulation, and optionally on quantum hardware.

The coursework is `notebooks/tutorial.ipynb`; it states the questions and marks.

## Setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Put the processed data from the course page in `data/svo-probes/processed/` (`all.pkl`, `image_embeddings.pt`).
Without it, `modules.data.load_svo_probes` rebuilds it from the original sources (about 10,000 image downloads, more
than an hour). Check the installation with `python -m pytest tests`.

## Layout
| file | contents |
|---|---|
| `modules/diagrams.py` | CCG parse trees -> string diagrams `(word, wires, types)`; the shared `ROOT` readout box |
| `modules/einsum.py` | `Recipe` (einsum string + `Op` per tensor) and `EinsumModel`, which contracts batches of recipes |
| `modules/classical.py` | `TNCompiler` (diagram -> tensor network, optional MPS), `TNModel`, `PCA` image projection |
| `modules/quantum.py` | gates, `CircuitBuilder`, `Ansatz` (diagram -> circuit), `VQCModel`, `AngleEncoder` image map |
| `modules/circuits.py` | recipes -> Qiskit circuits, compute–uncompute overlap circuits, exact and sampled overlaps |
| `modules/training.py` | cosine and fidelity similarities, `InfoNCE`, `choice_scores` |
| `modules/data.py` | SVO-Probes download/parse/embed/load, swap set, `ChoiceDataset` |
| `tests/` | checks that the torch contractions agree with Qiskit statevectors |
