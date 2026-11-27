"""Diagram -> quantum circuit recipe: a circuit per word, applied from the leaves of the diagram to its head."""
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import count

import opt_einsum as oe

from comp0267.compilation.einsum import Op, Recipe
from comp0267.compilation.gates import FIXED
from comp0267.symbolic.diagrams import Diagram, add_root, symbol


class CircuitBuilder:
    """Writes a circuit as a recipe. Each qubit carries an einsum label that changes after every gate."""

    def __init__(self) -> None:
        self.subscripts: list[str] = []
        self.ops: list[Op] = []
        self._labels = map(oe.get_symbol, count())

    def qubit(self) -> str:
        """A new qubit in state |0>; returns its label."""
        label = next(self._labels)
        self.subscripts.append(label)
        self.ops.append(Op(None, "0", (2,)))
        return label

    def postselect(self, label: str) -> None:
        """Post-select the qubit whose current label is ``label`` on 0 (it leaves the circuit)."""
        self.subscripts.append(label)
        self.ops.append(Op(None, "post0", (2,)))

    def apply(self, kind: str, wires: list[str], qubits: Sequence[int], name: str | None = None) -> None:
        """Apply gate ``kind`` to ``wires[q] for q in qubits`` (``wires`` is updated); ``name`` labels its angle."""
        if (name is None) != (kind in FIXED):
            raise ValueError(f"gate {kind!r}: fixed gates take no name, parametrised gates need one")
        outs = [next(self._labels) for _ in qubits]
        self.subscripts.append("".join(wires[q] for q in qubits) + "".join(outs))
        self.ops.append(Op(name, kind, (2,) * (2 * len(qubits))))
        for q, out in zip(qubits, outs):
            wires[q] = out

    def recipe(self, outputs: Sequence[str]) -> Recipe:
        """Finish the circuit: ``outputs`` are the labels of its output qubits."""
        return Recipe(",".join(self.subscripts) + "->" + "".join(outputs), tuple(self.ops))


def leaves_first(diagram: Diagram) -> Diagram:
    """The boxes ordered from the leaves (e.g. nouns) to the head that holds the output wire."""
    counts = Counter(w for _, wires, _ in diagram for w in wires)
    remaining = list(range(len(diagram)))
    order = [i for i in remaining if any(counts[w] == 1 for w in diagram[i][1])]
    remaining = [i for i in remaining if i not in order]
    seen = {w for i in order for w in diagram[i][1]}
    while remaining:  # breadth first from the head
        i = next((i for i in remaining if seen & set(diagram[i][1])), None)
        if i is None:
            raise ValueError("the diagram is not connected")
        order.append(i)
        remaining.remove(i)
        seen |= set(diagram[i][1])
    return [diagram[i] for i in reversed(order)]


class Ansatz(ABC):
    """Diagram -> Recipe, with ``qubits[type]`` qubits per wire, built from the leaves (e.g. nouns) to the head.

    A word acts on the qubits its arguments already hold, plus fresh qubits for the wires it creates; a word with as
    many new wires as arguments (e.g. a determiner) reuses its arguments' qubits. The wires left at the end, other
    than the output, are post-selected on 0.

    With ``root=False`` (default) the output is the head's open sentence wire; with ``root=True`` a shared ``ROOT``
    circuit maps it to ``qubits['out']`` output qubits, as in the classical model.
    """

    def __init__(self, qubits: Mapping[str, int], layers: int = 1, root: bool = False) -> None:
        self.qubits = dict(qubits)
        self.layers = layers
        self.root = root

    @abstractmethod
    def word_circuit(self, circuit: CircuitBuilder, wires: list[str], name: str) -> None:
        """Apply one word's gates to its qubits ``wires``. Name every angle ``f"{name}_..."``."""

    def __call__(self, diagram: Diagram) -> Recipe:
        counts = Counter(w for _, wires, _ in diagram for w in wires)
        if dangling := sorted(w for w, c in counts.items() if c == 1 and w != 0):
            raise ValueError(f"diagram wires {dangling} are open; only the output wire 0 may be")
        circuit = CircuitBuilder()
        qubits_of: dict[int, list[str]] = {}  # diagram wire -> its qubits
        for word, ccg_wires, types in leaves_first(add_root(diagram) if self.root else diagram):
            old = [(w, t) for w, t in zip(ccg_wires, types) if w in qubits_of]
            new = [(w, t) for w, t in zip(ccg_wires, types) if w not in qubits_of]
            if len(old) == len(new) and all(self.qubits[a] == self.qubits[b] for (_, a), (_, b) in zip(old, new)):
                # e.g. a determiner or an adjective: it maps its argument's qubits to its own new wire
                wires = [q for w, _ in old for q in qubits_of.pop(w)]
                owners = new
            else:  # e.g. a verb: it acts on its arguments' qubits and fresh qubits for its new wires
                wires = [q for w, t in zip(ccg_wires, types)
                         for q in (qubits_of[w] if w in qubits_of else [circuit.qubit() for _ in range(self.qubits[t])])]
                owners = list(zip(ccg_wires, types))
            self.word_circuit(circuit, wires, symbol(word, types))
            start = 0
            for w, t in owners:  # the gates relabelled the qubits
                qubits_of[w], start = wires[start:start + self.qubits[t]], start + self.qubits[t]
        outputs = qubits_of.pop(0)
        for qubits in qubits_of.values():
            for q in qubits:
                circuit.postselect(q)
        return circuit.recipe(outputs=outputs)
