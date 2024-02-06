import torch


from dataset import ChessBoardData, Vocab
from mingpt.model import GPT
from mingpt.trainer import Trainer


def create_model():
    model_config = GPT.get_default_config()
    model_config.model_type = "gpt2"
    model_config.vocab_size = Vocab.length()
    model_config.block_size = 1024
    model = GPT(model_config)

    return model


def main():
    model = create_model()
    dataset = ChessBoardData("./data/data.json", 100, 10)

    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4  # many possible options, see the file
    train_config.max_iters = 1000
    train_config.batch_size = 32

    trainer = Trainer(train_config, model, dataset)
    trainer.run()


if __name__ == "__main__":
    main()
