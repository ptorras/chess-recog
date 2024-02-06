import torch


from dataset import ChessBoardData, Vocab
from mingpt.model import GPT
from mingpt.trainer import Trainer

from torchmetrics.text import CharErrorRate

import torch


def create_model():
    model_config = GPT.get_default_config()
    model_config.model_type = "gpt2"
    model_config.vocab_size = Vocab.length()
    model_config.block_size = 1024
    model = GPT(model_config)

    return model


def on_batch_end_callback(val_dataloader, trainer):
    print({"loss": trainer.loss})

    if trainer.iter_num % 100 == 0:
        error = validate(val_dataloader, trainer)
        torch.save(trainer.model.state_dict(), "./saved_pp.pth")
        print({"validation_error": error})


def validate(val_dataloader, trainer) -> float:
    trainer.model.eval()
    cer = CharErrorRate()

    with torch.no_grad():
        for x, y in val_dataloader:
            x = x.to(trainer.device)
            y = y.to(trainer.device)
            in_seq = x[:, : val_dataloader.dataset.pad_size]
            out_seq = y[:, val_dataloader.dataset.pad_size :]
            out = trainer.model.generate(
                in_seq, val_dataloader.dataset.output_pad_size, do_sample=False
            )  # using greedy argmax, not sampling
            preds = [
                "".join(Vocab.detokenise(Vocab.unpad(x.cpu().detach().numpy())))
                for x in out
            ]
            target = [
                "".join(Vocab.detokenise(Vocab.unpad(x.cpu().detach().numpy())))
                for x in out_seq
            ]

            cer(preds, target)
            break
    trainer.model.train()

    return cer.compute()


def main():
    model = create_model()
    dataset = ChessBoardData("./data/data.json", 100, 10)
    val_dataset = ChessBoardData("./data/data.json", 100, 10)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=torch.utils.data.RandomSampler(
            val_dataset, replacement=True, num_samples=int(1e10)
        ),
        shuffle=False,
        pin_memory=True,
        batch_size=32,
        num_workers=8,
    )

    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4  # many possible options, see the file
    train_config.max_iters = 1000
    train_config.batch_size = 32

    trainer = Trainer(train_config, model, dataset)
    trainer.set_callback(
        "on_batch_end", lambda x: on_batch_end_callback(val_dataloader, x)
    )
    trainer.run()


if __name__ == "__main__":
    main()
