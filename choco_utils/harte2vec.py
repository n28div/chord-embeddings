"""

"""
from pathlib import Path

from lark import Lark, Tree


def decompose_harte(harte_chord: str) -> Tree:
    """

    Args:
        harte_chord:

    Returns:

    """
    assert ':' in harte_chord, 'The input chord is nota a valid Harte chord.'
    grammar = open(Path('harte.lark'))
    harte_parser = Lark(grammar)
    harte_tree = harte_parser.parse(harte_chord)
    return harte_tree


if __name__ == '__main__':
    decomposed = decompose_harte('C:7(3,*5,9)')
    print(decomposed.pretty())
