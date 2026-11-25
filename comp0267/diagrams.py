"""Sentences -> CCG parse trees -> string diagrams.

A diagram is a list of boxes ``(word, wires, types)``: one box per word, ``wires`` are integer wire labels and
``types`` the atomic grammatical type (``'n'``, ``'s'``, ...) of each wire. Two boxes sharing a wire label are
connected (contracted); wire ``0`` is the sentence output.
"""
import re
from collections import Counter
from collections.abc import Callable, Sequence
from itertools import count

from lambeq import BobcatParser
from lambeq.text2diagram import CCGRule, CCGTree, CCGType

from comp0267.assets import fetch

Box = tuple[str, list[int], list[str]]
Diagram = list[Box]

COMPOSITION_RULES = (CCGRule.FORWARD_COMPOSITION, CCGRule.BACKWARD_COMPOSITION,
                     CCGRule.FORWARD_CROSSED_COMPOSITION, CCGRule.BACKWARD_CROSSED_COMPOSITION)


def normalise(sentence: str) -> str:
    """Lower-case and strip punctuation."""
    return " ".join(re.sub(r"[^\w\s]", "", sentence).split()).lower()


def wire_types(ccg_type: CCGType) -> list[str]:
    """Atomic types of the wires of a (possibly complex) CCG type, left to right."""
    result, stack = [], [ccg_type]
    while stack:
        t = stack.pop()
        if not (t.is_over or t.is_under):
            result.append(t.name)
            continue
        arg = t.argument
        if arg.is_complex:  # a complex argument contributes its wires in reverse (its adjoint)
            arg = CCGType(result=arg.argument, direction=arg.direction, argument=arg.result)
        stack.extend([t.result, arg] if t.is_over else [arg, t.result])
    return result[::-1]


def tree_to_diagram(tree: CCGTree) -> Diagram:
    """Read a diagram off a CCG tree: each rule decides which wires its two children share."""
    fresh = count()

    def new(n: int) -> list[int]:
        return [next(fresh) for _ in range(n)]

    def fit(wires: list[int], n: int) -> list[int]:
        return wires[:n] + new(n - len(wires))

    def size(node: CCGTree) -> int:
        return len(wire_types(node.biclosed_type))

    diagram: Diagram = []
    stack = [(tree._resolved().collapse_noun_phrases(), new(1))]
    while stack:
        node, wires = stack.pop()
        rule = node.rule
        if rule == CCGRule.LEXICAL:
            diagram.append((node.text, wires, wire_types(node.biclosed_type)))
        elif rule == CCGRule.REMOVE_PUNCTUATION_LEFT:
            stack.append((node.right, wires))
        elif rule == CCGRule.REMOVE_PUNCTUATION_RIGHT:
            stack.append((node.left, wires))
        elif rule == CCGRule.UNARY:
            child = node.children[0]
            stack.append((child, wires + new(size(child) - len(wires))))
        elif rule == CCGRule.FORWARD_APPLICATION:  # X/Y  Y  ->  X
            n_l, n_r = size(node.left), size(node.right)
            shared = new(n_r)
            stack += [(node.left, fit(wires, n_l - n_r) + shared[::-1]), (node.right, shared)]
        elif rule == CCGRule.BACKWARD_APPLICATION:  # Y  X\Y  ->  X
            n_l, n_r = size(node.left), size(node.right)
            shared = new(n_l)
            stack += [(node.left, shared), (node.right, shared[::-1] + fit(wires, n_r - n_l))]
        elif rule in COMPOSITION_RULES:  # the children share the wires of the cancelled type Y
            n_l, n_r = size(node.left), size(node.right)
            n_y = (n_l + n_r - size(node)) // 2
            shared = new(n_y)
            if rule == CCGRule.FORWARD_COMPOSITION:
                x = fit(wires[:n_l - n_y], n_l - n_y)
                z = fit(wires[n_l - n_y:], n_r - n_y)
                stack += [(node.left, x + shared[::-1]), (node.right, shared + z)]
            elif rule == CCGRule.BACKWARD_COMPOSITION:
                z = fit(wires[:n_l - n_y], n_l - n_y)
                x = fit(wires[n_l - n_y:], n_r - n_y)
                stack += [(node.left, z + shared), (node.right, shared[::-1] + x)]
            elif rule == CCGRule.FORWARD_CROSSED_COMPOSITION:
                z = fit(wires[:n_r - n_y], n_r - n_y)
                x = fit(wires[n_r - n_y:], n_l - n_y)
                stack += [(node.left, x + shared[::-1]), (node.right, z + shared)]
            else:
                x = fit(wires[:n_r - n_y], n_r - n_y)
                z = fit(wires[n_r - n_y:], n_l - n_y)
                stack += [(node.left, shared + z), (node.right, shared[::-1] + x)]
        else:
            raise ValueError(f"CCG rule {rule} is not supported")
    return diagram


def add_root(diagram: Diagram) -> Diagram:
    """Attach a shared ``ROOT`` box that maps the open wires of the head word to one output wire of type ``out``.

    The head is the box holding every open wire (usually the verb). Its open wires are relabelled and fed into
    ``ROOT``, whose last wire is the sentence output ``0``. Every sentence with the same open types shares ``ROOT``.
    """
    counts = Counter(w for _, wires, _ in diagram for w in wires)
    open_wires = [w for w, c in counts.items() if c == 1]
    heads = [i for i, (_, wires, _) in enumerate(diagram) if set(open_wires) <= set(wires)]
    if not heads:
        raise ValueError("no single box holds all open wires")
    word, wires, types = diagram[heads[0]]
    relabel = {w: max(counts) + 1 + k for k, w in enumerate(open_wires)}
    open_types = [types[wires.index(w)] for w in open_wires]
    rooted = [(w, list(ws), list(ts)) for w, ws, ts in diagram]
    rooted[heads[0]] = (word, [relabel.get(w, w) for w in wires], list(types))
    return rooted + [("ROOT", list(relabel.values()) + [0], open_types + ["out"])]


def symbol(word: str, types: Sequence[str]) -> str:
    """Name of the parameters of a word: the same word with different types gets different parameters."""
    return f"{word}_{'@'.join(types)}"


def load_parser(device: str | None = None) -> BobcatParser:
    """The Bobcat CCG parser, with its model downloaded and cached on first use (lambeq's own server is gone)."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return BobcatParser(model_name_or_path=str(fetch("bobcat")), device=device, verbose="suppress")


def parse(sentences: Sequence[str], parser: BobcatParser | None = None,
          on_error: Callable[[str, Exception], None] | None = None) -> list[Diagram | None]:
    """Parse sentences into (unrooted) diagrams; ``None`` where parsing or reading the diagram fails."""
    parser = parser or load_parser()
    trees = parser.sentences2trees([normalise(s) for s in sentences], suppress_exceptions=True)
    diagrams = []
    for sentence, tree in zip(sentences, trees):
        try:
            diagrams.append(None if tree is None else tree_to_diagram(tree))
        except ValueError as err:
            if on_error:
                on_error(sentence, err)
            diagrams.append(None)
    return diagrams
