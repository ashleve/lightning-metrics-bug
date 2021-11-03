import torch

from pytorch_lightning import LightningModule, Trainer
from mnist_datamodule import MNISTDataModule
from simple_dense_net import SimpleDenseNet
from torchmetrics.classification.accuracy import Accuracy


class MNISTModel(LightningModule):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = SimpleDenseNet(hparams=self.hparams)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.val_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx: int):
        loss, preds, targets = self.step(batch)
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, preds, targets = self.step(batch)

        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_epoch_end(self, outputs):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.log("val/acc_2", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)


def run():
    datamodule = MNISTDataModule()

    model = MNISTModel()
    trainer = Trainer(
        max_epochs=3,
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    run()
