import torch.utils.data as D
import json
import string

from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from numpy.typing import ArrayLike


class Vocab:
    """Implements switching back and forth between tokens."""

    TOKEN2INDEX = {token: ii for ii, token in enumerate(string.printable)}
    INDEX2TOKEN = {ii: token for ii, token in enumerate(string.printable)}

    @classmethod
    def tokenise(cls, inp: str) -> List[int]:
        return [cls.TOKEN2INDEX[x] for x in inp]

    @classmethod
    def pad(cls, inp: List[int], pad_len: int) -> ArrayLike:
        padded = np.zeros(pad_len, dtype=int)
        padded[: len(inp)] = inp

        return padded

    @classmethod
    def length(cls) -> int:
        return len(cls.TOKEN2INDEX)


class ChessBoardData(D.Dataset):
    def __init__(self, path: Path, pad_size: int, output_pad_size: int) -> None:
        self.path = path
        self.pad_size = pad_size
        self.output_pad_size = output_pad_size

        with open(path, "r", encoding="utf8") as f_in:
            data = json.load(f_in)

        self.data = [
            {
                "fen": Vocab.tokenise(x["fen"]),
                "real": Vocab.tokenise(x["real"]),
                "pre": Vocab.tokenise(x["pre"]),
                "og_fen_len": len(x["fen"]),
                "og_real_len": len(x["real"]),
                "og_predicted_len": len(x["pre"]),
            }
            for x in data
            if len(x["fen"]) + len(x["pre"]) <= self.pad_size - 2
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[ArrayLike, ArrayLike]:
        sample = self.data[index]
        padded = Vocab.pad(
            sample["fen"] + sample["pre"] + sample["real"],
            self.pad_size + self.output_pad_size,
        )
        print(padded)

        return padded[:-1], padded[1:]
