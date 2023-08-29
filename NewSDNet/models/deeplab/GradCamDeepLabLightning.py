from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import psutil
import wandb
from pathlib import Path
from NewSDNet.utils.losses import *
import random


# NOTE assume model output of shape (batch,2classes,H,W) with logits output
# --> for cross entropy loss


class GradCamDeepLabLightning(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float,
        img_logger: pl.Callback,
        batch_size: int,
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
        self.batch_size = batch_size
        self.num_classes = 5
        # Create loss module
        self.loss_segmentation = dice_loss()  # nn.CrossEntropyLoss()
        self.lr = lr
        self.img_logger = img_logger
        self.save_path = save_path
        self.max_dice_train = -1
        self.train_table: list[list[Any]] = []
        self.columns_train: list[str] = [
            "centre",
            "image",
            "saliency map",
            "gradcam prediction",
            "ground truth",
            "prediction",
        ]
        self.columns_val: list[str] = [
            "centre",
            "dice score",
            "image",
            "ground truth",
            "prediction",
        ]
        self.val_table = []
        self.in_test_table = []
        self.out_test_table = []
        self.columns_test = [
            "centre",
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
        return optimizer

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        # "batch" is the output of the training data loader.
        imgs, labels, _, saliency_maps = batch
        prob = random.random()
        # print(f"\nPROB: {prob}")
        if prob < 0.60:
            # print(f"\nSONO NEL IF SALIENCY / IF RANDOM.CHOICE = 1")
            # print(f"\nSONO NEL NOT NON DEL FORWARD DELLE SALIENCY!!!\n")
            input_to_segmentor = (
                imgs * saliency_maps[:, :3, :, :]
            )  # this is a change w.r.t original architecture
            # print(
            #     f"\nGUARDO SE A_OUT E INPUT TO SEGMENTOR SONO DIVERSI: {torch.unique(imgs[imgs != input_to_segmentor], return_counts=True)}"
            # )
        else:
            # print(f"\nSONO NEL IF SALIENCY / IF RANDOM.CHOICE = 0")
            input_to_segmentor = imgs

        segmentation_preds = self.model(
            input_to_segmentor
        )  # features shape = [8, 512, 32, 32]
        loss_seg = self.loss_segmentation(segmentation_preds, labels)

        # compute the dice score
        dice_score = torchmetrics.functional.dice(
            segmentation_preds, labels, zero_division=1, num_classes=2, ignore_index=0
        )

        self.log("train_dice", dice_score, on_step=False, on_epoch=True)
        self.log("train_segmentation_loss", loss_seg, on_step=False, on_epoch=True)
        self.log("RAM", psutil.virtual_memory().percent, on_step=True)
        return {
            "loss": loss_seg,
            "segmentation_preds": segmentation_preds,
            "input_to_segmentor": input_to_segmentor,
        }  # Return tensor to call ".backward" on

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        imgs, labels, centres, saliency_maps = batch
        segmentation_preds = outputs["segmentation_preds"]
        input_to_segmentor = outputs["input_to_segmentor"]

        dices = torchmetrics.functional.dice(
            segmentation_preds,
            labels,
            zero_division=1,
            num_classes=2,
            ignore_index=0,
        )
        segmentation_preds_to_plot = torch.argmax(segmentation_preds, 1)
        if self.current_epoch == (self.trainer.max_epochs - 1):
            if batch_idx % 10 == 0:
                images = [
                    [
                        centre.to(torch.float),
                        img,
                        saliency_map[0, ...].float(),
                        input_to_seg.float(),
                        label.float(),
                        segmentation_pred.float(),
                    ]
                    for centre, img, saliency_map, input_to_seg, label, segmentation_pred in zip(
                        centres,
                        imgs,
                        saliency_maps,
                        input_to_segmentor,
                        labels,
                        segmentation_preds_to_plot,
                    )
                ]
                self.train_table.extend(images)

    def validation_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        with torch.enable_grad():
            imgs, labels, _ = batch

            segmentation_preds = self.model(imgs)  # features shape = [8, 512, 32, 32]
            loss_seg = self.loss_segmentation(segmentation_preds, labels)

            # compute the dice score
            dice_score = torchmetrics.functional.dice(
                segmentation_preds,
                labels,
                zero_division=1,
                num_classes=2,
                ignore_index=0,
            )

            self.log("val_dice", dice_score, on_step=False, on_epoch=True)
            self.log("val_segmentation_loss", loss_seg, on_step=False, on_epoch=True)
            self.log("RAM", psutil.virtual_memory().percent, on_step=True)
            return segmentation_preds, dice_score

    def on_validation_batch_end(
        self,
        outputs: Optional[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        imgs, labels, centre = batch

        segmentation_preds, dice = outputs

        segmentation_preds_to_plot = torch.argmax(segmentation_preds, 1)

        if self.current_epoch == (self.trainer.max_epochs - 1):
            if batch_idx % 10 == 0:
                images = [
                    [
                        centre.to(torch.float),
                        dice,
                        img,
                        label.float(),
                        segmentation_pred.float(),
                    ]
                    for img, label, segmentation_pred in zip(
                        imgs,
                        labels,
                        segmentation_preds_to_plot,
                    )
                ]
                self.val_table.extend(images)

    def on_train_end(self) -> None:
        train_table_wandb = []
        for row in self.train_table:
            table_wand = [
                row[0],
                wandb.Image(row[1]),
                wandb.Image(row[2]),
                wandb.Image(row[3]),
                wandb.Image(row[4]),
                wandb.Image(row[5]),
            ]
            train_table_wandb.append(table_wand)

        val_table_wandb = []
        for row in self.val_table:
            table_wand = [
                row[0],
                row[1],
                wandb.Image(row[2]),
                wandb.Image(row[3]),
                wandb.Image(row[4]),
            ]
            val_table_wandb.append(table_wand)
        self.img_logger.log_table(
            key="training predictions",
            columns=self.columns_train,
            data=train_table_wandb,
        )
        self.img_logger.log_table(
            key="validation predictions",
            columns=self.columns_val,
            data=val_table_wandb,
        )

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        imgs, labels, _ = batch
        segmentation_preds = self.model(imgs)  # features shape = [8, 512, 32, 32]

        # compute the dice score
        dice_score = torchmetrics.functional.dice(
            segmentation_preds, labels, zero_division=1, num_classes=2, ignore_index=0
        )
        recall_score = torchmetrics.functional.classification.binary_recall(
            torch.argmax(segmentation_preds, 1),
            labels,
            threshold=0.5,
            multidim_average="global",
            ignore_index=0,
            validate_args=True,
        )
        accuracy_score = torchmetrics.functional.classification.binary_accuracy(
            torch.argmax(segmentation_preds, 1),
            labels,
            threshold=0.5,
            multidim_average="global",
            ignore_index=None,
            validate_args=True,
        )

        self.log_dict(
            {
                f"{dataloader_idx}_test_dice": dice_score,
                f"{dataloader_idx}_recall": recall_score,
                f"{dataloader_idx}_accuracy": accuracy_score,
            }
        )
        return segmentation_preds

    def on_test_batch_end(
        self,
        outputs: Optional[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        imgs, labels, centre = batch
        segmentation_preds = outputs
        dice = torchmetrics.functional.dice(
            segmentation_preds, labels, zero_division=1, num_classes=2, ignore_index=0
        )
        recall_score = torchmetrics.functional.classification.binary_recall(
            torch.argmax(segmentation_preds, 1),
            labels,
            threshold=0.5,
            multidim_average="global",
            ignore_index=0,
            validate_args=True,
        )
        accuracy_score = torchmetrics.functional.classification.binary_accuracy(
            torch.argmax(segmentation_preds, 1),
            labels,
            threshold=0.5,
            multidim_average="global",
            ignore_index=None,
            validate_args=True,
        )
        preds_plot = torch.argmax(segmentation_preds, 1)
        images = [
            [
                centre.to(torch.float),
                dice,
                recall_score,
                accuracy_score,
                img,
                lbl.float(),
                pred.float(),
            ]
            for img, lbl, pred in zip(imgs, labels, preds_plot)
        ]

        if dataloader_idx == 0:
            self.in_test_table.extend(images)
        # else:
        #     self.out_test_table.extend(images)

    def on_test_end(self) -> None:
        in_test_table_wandb = []
        for row in self.in_test_table:
            table_wand = [
                row[0],
                row[1],
                row[2],
                row[3],
                wandb.Image(row[4]),
                wandb.Image(row[5]),
                wandb.Image(row[6]),
            ]
            in_test_table_wandb.append(table_wand)

        # out_test_table_wandb = []
        # for row in self.out_test_table:
        #     table_wand = [
        #         row[0],
        #         row[1],
        #         row[2],
        #         row[3],
        #         wandb.Image(row[4]),
        #         wandb.Image(row[5]),
        #         wandb.Image(row[6]),
        #     ]
        #     out_test_table_wandb.append(table_wand)

        self.img_logger.log_table(
            key="in distribution test set predictions",
            columns=self.columns_test,
            data=in_test_table_wandb,
        )

        # self.img_logger.log_table(
        #     key="out distribution test set predictions",
        #     columns=self.columns_test,
        #     data=out_test_table_wandb,
        # )
