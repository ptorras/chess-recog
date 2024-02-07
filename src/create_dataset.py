import json

import chess
import chess.pgn

path_file = "/home/ptorras/Documents/Datasets/LichessDatabase/lichess_db_standard_rated_2013-01.pgn"

begin_seq = "[Termination"


def eng2cat(original_string):
    piece_names_eng = ["N", "Q", "K", "P", "B", "R"]
    piece_names_eng_ = ["n", "q", "k", "p", "b", "r"]
    piece_names_cat = ["C", "D", "R", "P", "A", "T"]
    piece_names_cat_ = ["c", "d", "r", "p", "a", "t"]

    for letter in range(len(piece_names_eng)):
        original_string = original_string.replace(
            piece_names_eng[letter], piece_names_cat[letter]
        )
        original_string = original_string.replace(
            piece_names_eng_[letter], piece_names_cat_[letter]
        )
    return original_string


# with open(path_file, "r+") as f:
#     content = f.read()
#     replaced_content = re.sub(r"Opening", "pepino", content)
#     f.seek(0)
#     f.write(replaced_content)
#     f.truncate()


cont_rows = 0
play_full = []
with open(path_file, "r") as f:
    for row in f:
        cont_rows += 1
        if begin_seq in row:
            play_full.append(cont_rows + 2)

# play_full = [0]
data = []
with open("/home/ptorras/Documents/Datasets/LichessDatabase/data_full.json", "w") as f:
    with open(path_file) as pgn:
        for i in range(len(play_full)):
            #         for i in range(1):
            game = chess.pgn.read_game(pgn)
            board = game.board()
            for move in game.mainline_moves():
                try:
                    fen = eng2cat(board.fen())  # board.fen()
                    real = eng2cat(board.san(move))  # board.san(move)
                    pre = eng2cat(board.san(move))  # board.san(move)# his can be noise
                    data.append({"fen": fen, "real": real, "pre": pre})
                except:
                    print(f"ILEGAL MOVE", move)
                board.push(move)
    json.dump(data, f)
