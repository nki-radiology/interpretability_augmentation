from typing import Any, Optional
import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import psutil
import wandb
from pathlib import Path
from NewSDNet.utils.losses import *


# NOTE assume model output of shape (batch,2classes,H,W) with logits output
# --> for cross entropy loss


class UNetLightning(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float,
        img_logger: pl.Callback,
        save_path: Path,
    ):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        self.model = model
        # Create loss module
        self.loss_module = dice_loss()  # nn.CrossEntropyLoss()
        self.lr = lr
        self.img_logger = img_logger
        self.save_path = save_path
        self.max_dice_train = -1
        self.table = []
        self.in_test_table = []
        self.out_test_table = []
        self.columns = ["dice score", "image", "ground_truth", "prediction"]
        self.columns_test = [
            "test",
            "dice score",
            "recall",
            "accuracy",
            "image",
            "ground_truth",
            "prediction",
        ]
        self.out_dice = []

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # AdamW is Adam with a correct implementation of weight decay (see here for details: https://arxiv.org/pdf/1711.05101.pdf)
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        # scheduler = optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[100, 150], gamma=0.1
        # )
        return optimizer

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        # "batch" is the output of the training data loader.
        imgs, labels, _ = batch
        features, preds = self.model(imgs)  # features shape = [8, 512, 32, 32]
        loss = self.loss_module(preds, labels)
        # compute the dice score
        dice_score = torchmetrics.functional.dice(
            preds, labels, zero_division=1, num_classes=2, ignore_index=0
        )

        self.log("train_dice", dice_score, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("RAM", psutil.virtual_memory().percent, on_step=True)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels, _ = batch
        _, preds = self.model(imgs)
        loss = self.loss_module(preds, labels)

        # compute dice score
        dice_score = torchmetrics.functional.dice(
            preds, labels, zero_division=1, num_classes=2, ignore_index=0
        )

        # By default logs it per epoch (weighted average over batches)
        self.log_dict({"val_dice": dice_score, "val_loss": loss})
        return preds

    def on_validation_batch_end(
        self,
        outputs: Optional[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        imgs, labels, _ = batch
        dice = torchmetrics.functional.dice(
            outputs, labels, zero_division=1, num_classes=2, ignore_index=0
        )
        preds_plot = torch.argmax(outputs, 1)

        if self.current_epoch == (self.trainer.max_epochs):
            if batch_idx % 20 == 0:
                images = [
                    [
                        dice,
                        wandb.Image(img),
                        wandb.Image(lbl.float()),
                        wandb.Image(pred.float()),
                    ]
                    for img, lbl, pred in zip(imgs, labels, preds_plot)
                ]
                self.table.extend(images)

    def on_train_end(self) -> None:
        self.img_logger.log_table(
            key="validation predictions", columns=self.columns, data=self.table
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, labels, _ = batch
        _, preds = self.model(imgs)

        # compute dice score
        dice_score = torchmetrics.functional.dice(
            preds, labels, zero_division=1, num_classes=2, ignore_index=0
        )
        recall_score = torchmetrics.functional.classification.binary_recall(
            torch.argmax(preds, 1),
            labels,
            threshold=0.5,
            multidim_average="global",
            ignore_index=0,
            validate_args=True,
        )
        accuracy_score = torchmetrics.functional.classification.binary_accuracy(
            torch.argmax(preds, 1),
            labels,
            threshold=0.5,
            multidim_average="global",
            ignore_index=None,
            validate_args=True,
        )

        # By default logs it per epoch (weighted average over batches)

        self.log_dict(
            {
                f"{dataloader_idx}_test_dice": dice_score,
                f"{dataloader_idx}_recall": recall_score,
                f"{dataloader_idx}_accuracy": accuracy_score,
            }
        )
        return preds

    def on_test_batch_end(
        self,
        outputs: Optional[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        imgs, labels, centre = batch
        dice = torchmetrics.functional.dice(
            outputs, labels, zero_division=1, num_classes=2, ignore_index=0
        )
        recall_score = torchmetrics.functional.classification.binary_recall(
            torch.argmax(outputs, 1),
            labels,
            threshold=0.5,
            multidim_average="global",
            ignore_index=0,
            validate_args=True,
        )
        accuracy_score = torchmetrics.functional.classification.binary_accuracy(
            torch.argmax(outputs, 1),
            labels,
            threshold=0.5,
            multidim_average="global",
            ignore_index=None,
            validate_args=True,
        )
        preds_plot = torch.argmax(outputs, 1)
        images = [
            [
                centre.to(torch.float),
                dice,
                recall_score,
                accuracy_score,
                wandb.Image(img),
                wandb.Image(lbl.float()),
                wandb.Image(pred.float()),
            ]
            for img, lbl, pred in zip(imgs, labels, preds_plot)
        ]

        if dataloader_idx == 0:
            self.in_test_table.extend(images)
        else:
            self.out_test_table.extend(images)

    def on_test_end(self) -> None:
        self.img_logger.log_table(
            key="in distribution test set predictions",
            columns=self.columns_test,
            data=self.in_test_table,
        )

        self.img_logger.log_table(
            key="out distribution test set predictions",
            columns=self.columns_test,
            data=self.out_test_table,
        )
