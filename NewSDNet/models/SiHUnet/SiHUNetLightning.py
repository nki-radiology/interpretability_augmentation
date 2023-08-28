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
from NewSDNet.utils.losses import GeneralizedCELoss, FocalLoss
import torch.nn.functional as F

# NOTE assume model output of shape (batch,2classes,H,W) with logits output
# --> for cross entropy loss


class SiHUNetLightning(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float,
        img_logger: pl.Callback,
        batch_size: int,
        save_path: Path,
        q: float,
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
        self.q = q
        # Create loss module
        self.loss_segmentation = nn.CrossEntropyLoss()
        self.loss_reconstruction = nn.L1Loss()
        self.generalized_crossentropy = GeneralizedCELoss(self.q)
        self.lr = lr
        self.img_logger = img_logger
        self.save_path = save_path
        self.max_dice_train = -1
        self.train_table: list[list[Any]] = []
        self.columns_train: list[str] = [
            "centre",
            "image",
            "biased segmentation",
            "unbiased segmentation",
            "ground truth",
            "reconstruction",
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
            "image",
            "ground_truth",
            "prediction",
        ]
        self.out_dice = []

    def forward(self, imgs, script_type="validation"):
        # Forward function that is run when visualizing the graph
        return self.model(imgs, script_type)

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
        (
            unbiased_segmentation_logits,
            biased_segmentation_logits,
            reconstruction_logits,
        ) = self.model(imgs, script_type="training")

        # Compute reconstruction loss
        reco_loss = self.loss_reconstruction(reconstruction_logits, imgs)
        # Consider turning the unet in a VAE and also have KL divergence

        # Compute reweighting
        prob_biased = F.softmax(biased_segmentation_logits, dim=1)
        print(
            f"\nShape di prob_b (quindi shape dei logits dopo il softmax): {prob_biased.shape}"
        )
        print(f"\nValori di prob_b: {prob_biased}")
        print(f"\n Shape di labels: {labels.shape}")
        print(f"\n Labels: {labels}")
        y_hat_biased = torch.gather(
            prob_biased, 1, torch.unsqueeze(labels, dim=1)
        ).detach()  # I don't think the unsqueeze is necessary
        print(f"\ny_hat_biased shape: {y_hat_biased.shape}")
        print(f"\n y_hat_biased: {y_hat_biased}")
        loss_weight = (1 - y_hat_biased) ** self.q
        real_diff_score = loss_weight.detach()
        logits_for_weight = biased_segmentation_logits.view(
            biased_segmentation_logits.size(0), biased_segmentation_logits.size(1), -1
        )  # N,C,H,W => N,C,H*W
        print(f"\nLOGITS FOR WEIGHT 1: {logits_for_weight.shape}")
        logits_for_weight = logits_for_weight.transpose(1, 2)  # N,C,H*W => N,H*W,C
        print(f"\nLOGITS FOR WEIGHT 2: {logits_for_weight.shape}")
        logits_for_weight = (
            logits_for_weight.contiguous().view(-1, logits_for_weight.size(2)).detach()
        ).detach()  # N,H*W,C => N*H*W,C
        print(f"\nLOGITS FOR WEIGHT 3: {logits_for_weight.shape}")
        targets_for_weight = labels.view(-1, 1)
        print(f"\nTARGETS FOR WEIGHT: {targets_for_weight.shape}")

        p = F.log_softmax(logits_for_weight)
        print(f"\nP: {p.shape}")
        assert p.mean().item() is not None
        Yg = torch.gather(p, 1, targets_for_weight)  # not sure this unsqueeze is needed

        # modify gradient of cross entropy
        Yg = Yg.view(-1).detach()
        loss_weight = (1 - Yg) ** self.q
        print(f"\nLOSS_WEIGHT: {loss_weight.shape}")
        real_diff_score = loss_weight.detach()
        focal_test = FocalLoss()
        focal = focal_test(biased_segmentation_logits, labels)
        print(
            f"\nTYPE DI LOSS_UNBIASED: {F.cross_entropy(unbiased_segmentation_logits, labels, reduction='mean').type}"
        )
        print(f"\nTYPE DI REAL_DIFF_SCORE: {real_diff_score.type}")
        # Calculate and weigh segmentation losses
        loss_unbiased = (
            F.cross_entropy(unbiased_segmentation_logits, labels, reduction="none")
            * real_diff_score
        )
        print(f"\nLOSS UNBIASED PRIMA DEL PRIMO MEAN: {loss_unbiased.shape}")
        loss_unbiased = loss_unbiased.mean()
        print(f"\nLOSS UNBIASED DOPO IL PRIMO MEAN: {loss_unbiased.shape}")
        loss_unbiased = loss_unbiased.mean()
        print(f"\nLOSS UNBIASED DOPO IL SECONDO MEAN: {loss_unbiased.shape}")
        print(f"\nShape di loss_unbiased: {loss_unbiased.shape}")
        loss_biased = self.generalized_crossentropy(biased_segmentation_logits, labels)
        print(f"\nShape di loss_biased: {loss_biased.shape}")

        loss = (loss_unbiased + loss_biased + reco_loss) / 3
        # compute the dice score
        dice_score_unbiased = torchmetrics.functional.dice(
            unbiased_segmentation_logits,
            labels,
            zero_division=1,
            num_classes=2,
            ignore_index=0,
        )
        dice_score_biased = torchmetrics.functional.dice(
            biased_segmentation_logits,
            labels,
            zero_division=1,
            num_classes=2,
            ignore_index=0,
        )

        self.log(
            "train_dice_unbiased", dice_score_unbiased, on_step=False, on_epoch=True
        )
        self.log("train_dice_biased", dice_score_biased, on_step=False, on_epoch=True)
        self.log("total_train_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "train_segmentation_loss_biased", loss_biased, on_step=False, on_epoch=True
        )
        self.log(
            "train_segmentation_loss_unbiased",
            loss_unbiased,
            on_step=False,
            on_epoch=True,
        )
        self.log("train_recontruction_loss", reco_loss, on_step=False, on_epoch=True)
        self.log("RAM", psutil.virtual_memory().percent, on_step=True)
        return {
            "loss": loss,
            "segmentation_preds_biased": biased_segmentation_logits,
            "segmentation_preds_unbiased": biased_segmentation_logits,
            "reconstruction": reconstruction_logits,
        }  # Return tensor to call ".backward" on

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        imgs, labels, centres = batch
        segmentation_preds_biased = outputs["segmentation_preds_biased"]
        segmentation_preds_unbiased = outputs["segmentation_preds_unbiased"]
        reconstruction = outputs["reconstruction"]

        biased_segmentation_to_plot = torch.argmax(segmentation_preds_biased, 1)
        unbiased_segmentation_to_plot = torch.argmax(segmentation_preds_unbiased, 1)
        if self.current_epoch == (self.trainer.max_epochs - 1):
            if batch_idx % 10 == 0:
                images = [
                    [
                        centre.to(torch.float),
                        img,
                        biased_seg.float(),
                        unbiased_seg.float(),
                        label.float(),
                        reco.float(),
                    ]
                    for centre, img, biased_seg, unbiased_seg, label, reco in zip(
                        centres,
                        imgs,
                        biased_segmentation_to_plot,
                        unbiased_segmentation_to_plot,
                        labels,
                        reconstruction,
                    )
                ]
                self.train_table.extend(images)

    def validation_step(self, batch, batch_idx):
        # "batch" is the output of the validation data loader.
        imgs, labels, _ = batch
        segmentation_preds = self.model(
            imgs, script_type="validation"
        )  # features shape = [8, 512, 32, 32]
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
        self.log("val_loss", loss_seg, on_step=False, on_epoch=True)
        self.log("RAM", psutil.virtual_memory().percent, on_step=True)
        return (
            loss_seg,
            segmentation_preds,
            dice_score,
        )

    def on_validation_batch_end(
        self,
        outputs: Optional[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        imgs, labels, centre = batch
        (_, segmentation_preds, dice_score) = outputs

        segmentation_preds_to_plot = torch.argmax(segmentation_preds, 1)

        if self.current_epoch == (self.trainer.max_epochs - 1):
            if batch_idx % 10 == 0:
                images = [
                    [
                        centre.to(torch.float),
                        dice_score,
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

    def test_step(self, batch, batch_idx, dataloader_idx):
        imgs, labels, _ = batch
        segmentation_preds = self.model(
            imgs, script_type="testing"
        )  # features shape = [8, 512, 32, 32]

        # compute the dice score
        dice_score = torchmetrics.functional.dice(
            segmentation_preds, labels, zero_division=1, num_classes=2, ignore_index=0
        )
        self.log_dict(
            {
                f"{dataloader_idx}_test_dice": dice_score,
            }
        )
        return segmentation_preds, dice_score

    def on_test_batch_end(
        self,
        outputs: Optional[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        imgs, labels, centre = batch
        segmentation_preds, dice_score = outputs

        preds_plot = torch.argmax(segmentation_preds, 1)
        images = [
            [
                centre.to(torch.float),
                dice_score,
                img,
                lbl.float(),
                pred.float(),
            ]
            for img, lbl, pred in zip(imgs, labels, preds_plot)
        ]

        if dataloader_idx == 0:
            self.in_test_table.extend(images)
        else:
            self.out_test_table.extend(images)

    def on_test_end(self) -> None:
        in_test_table_wandb = []
        for row in self.in_test_table:
            table_wand = [
                row[0],
                row[1],
                wandb.Image(row[2]),
                wandb.Image(row[3]),
                wandb.Image(row[4]),
            ]
            in_test_table_wandb.append(table_wand)

        out_test_table_wandb = []
        for row in self.out_test_table:
            table_wand = [
                row[0],
                row[1],
                wandb.Image(row[2]),
                wandb.Image(row[3]),
                wandb.Image(row[4]),
            ]
            out_test_table_wandb.append(table_wand)

        self.img_logger.log_table(
            key="in distribution test set predictions",
            columns=self.columns_test,
            data=in_test_table_wandb,
        )

        self.img_logger.log_table(
            key="out distribution test set predictions",
            columns=self.columns_test,
            data=out_test_table_wandb,
        )
