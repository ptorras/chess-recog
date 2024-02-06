import chess
from argparse import Namespace, ArgumentParser

def main(args: Namespace) -> None:
    ...


 def setup() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("img_path", type=Path, help="path to the image of the game.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(setup())
