import torch.utils.data as D
import json
import string

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from numpy import ArrayLike


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
                "predicted": Vocab.tokenise(x["predicted"]),
                "og_fen_len": len(x["fen"]),
                "og_real_len": len(x["real"]),
                "og_predicted_len": len(x["predicted"]),
            }
            for x in data
            if len(x["fen"]) + len(x["predicted"]) <= self.pad_size - 2
        ]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.data[index]
        return {
            "input": Vocab.pad(sample["fen"] + sample["predicted"], self.pad_size),
            "gt": Vocab.pad(sample["real"], self.output_pad_size),
        }
