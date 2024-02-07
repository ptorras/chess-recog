import json
import random
import string
from pathlib import Path
from string import ascii_letters, digits, printable
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch.utils.data as D
from numpy.typing import ArrayLike


class Vocab:
    """Implements switching back and forth between tokens."""

    TOKEN2INDEX = {token: ii for ii, token in enumerate("\r" + string.printable)}
    INDEX2TOKEN = {ii: token for ii, token in enumerate("\r" + string.printable)}

    @classmethod
    def tokenise(cls, inp: str) -> List[int]:
        return [cls.TOKEN2INDEX[x] for x in inp]

    @classmethod
    def pad(cls, inp: List[int], pad_len: int) -> ArrayLike:
        padded = np.zeros(pad_len, dtype=int)
        padded[: len(inp)] = inp

        return padded

    @classmethod
    def unpad(cls, inp: ArrayLike) -> List[int]:
        return [x for x in inp if x != 0]

    @classmethod
    def detokenise(cls, inp: List[int]) -> str:
        return "".join(cls.INDEX2TOKEN[x] for x in inp)

    @classmethod
    def length(cls) -> int:
        return len(cls.TOKEN2INDEX)


def augment_output(val: str) -> str:
    values = list(val)
    if random.random() > 0.5:
        values = random_insert(values)
    if random.random() > 0.5:
        values = random_substitution(values)
    if random.random() > 0.5:
        values = random_case(values)

    return "".join(values)


SIMILAR = {
    "C": ["(", "c", "/", "["],
    "d": ["4", "9", "b", "p"],
}


def similar_elements(val: List[str]) -> List[str]:
    ...


def random_insert(val: List[str]) -> List[str]:
    index = random.randint(0, len(val) + 1)
    rchindex = random.randint(0, len(printable))
    randchar = printable[rchindex]
    val.insert(index, randchar)
    return val


def random_substitution(val: List[str]) -> List[str]:
    index = random.randint(0, len(val))
    rchindex = random.randint(0, len(printable))
    val[index] = printable[rchindex]
    return val


def random_case(val: List[str]) -> List[str]:
    for elm in range(len(val)):
        if random.random() > 0.35:
            val[elm] = val[elm].swapcase()
    return val


class ChessBoardData(D.Dataset):
    def __init__(
        self,
        path: Path,
        pad_size: int,
        output_pad_size: int,
        augmentation: Callable[[str], str] | None = None,
    ) -> None:
        self.path = path
        self.pad_size = pad_size
        self.output_pad_size = output_pad_size
        self.augmentation = augmentation

        with open(path, "r", encoding="utf8") as f_in:
            data = json.load(f_in)

        self.data = [
            {
                "fen": Vocab.tokenise(x["fen"]),
                "real": Vocab.tokenise(x["real"]),
                "pre": x["pre"],
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
        padded = np.zeros(self.pad_size + self.output_pad_size, dtype=int)
        pred = sample["pre"]
        if self.augmentation is not None:
            pred = self.augmentation(pred)
        pred_string = Vocab.tokenise(pred)
        padded[: self.pad_size] = Vocab.pad(
            sample["fen"] + pred_string,
            self.pad_size,
        )
        padded[self.pad_size :] = Vocab.pad(
            sample["real"],
            self.output_pad_size,
        )

        return padded[:-1], padded[1:]
