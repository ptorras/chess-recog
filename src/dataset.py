import torch.utils.data as D
import json

from pathlib import Path
from typing import Dict, Any


PIECE_NAMES = {"K", "Q", "R", "B", "N", "P"}


class FenVocab:
    FEN_TOKENS = {x for x in range(ord())}

    def __init__(self) -> None:
        ...


class PgnVocab:
    def __init__(self) -> None:
        ...


class UciVocab:
    def __init__(self) -> None:
        ...


class ChessBoardData(D.Dataset):
    def __init__(self, path: Path) -> None:
        ...

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ...
