from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from NewSDNet.utils.losses import *
from NewSDNet.utils.init_weight import *
import psutil
import wandb


class SDNetLightning(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float,
        img_logger: pl.Callback,
        batch_size: int,
        weight_init: str,
        charbonnier: int,
        reco_w: float,
        kl_w: float,
        dice_w: float,
        regress_w: float,
        focal_w: float,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.img_logger = img_logger
        self.b_images = torch.zeros(batch_size, 3, 512, 512)
        self.b_masks = torch.zeros(batch_size, 2, 512, 512)
        self.collapsed_b_masks = torch.zeros(batch_size, 1, 512, 512)
        self.weight_init = weight_init
        self.charbonnier = charbonnier
        self.reco_w = reco_w
        self.kl_w = kl_w
        self.dice_w = dice_w
        self.regress_w = regress_w
        self.focal_w = focal_w
        self.l1_distance = nn.L1Loss()
        self.anatomy_train = []
        self.anatomy_val = []
        self.anatomy_in_test = []
        self.anatomy_out_test = []
        self.columns_anatomy = [
            "channel 0",
            "channel 1",
            "channel 2",
            "channel 3",
            "channel 4",
            "channel 5",
            "channel 6",
            "channel 7",
        ]
        self.train_table = []
        self.val_table = []
        self.in_test_table = []
        self.out_test_table = []
        self.columns_table = [
            "reconstruction",
            "ground_truth",
            "prediction",
            "image",
        ]
        self.columns_table_val = [
            "dice",
            "reconstruction",
            "ground_truth",
            "prediction",
            "image",
        ]
        self.columns_table_test = [
            "centre",
            "dice",
            "recall",
            "accuracy",
            "reconstruction",
            "ground_truth",
            "prediction",
            "image",
        ]
        initialize_weights(self.model, self.weight_init)

    def forward(self, imgs, script_type="validation"):
        return self.model(imgs)

    def configure_optimizers(self) -> Any:
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        imgs, labels, _ = batch

        # Add channel for background to masks:
        for i in range(labels.shape[0]):
            cmask = labels[i].unsqueeze(0)
            logical_or = torch.sum(cmask, dim=0)
            tmask_0 = 1 - logical_or
            tmask = torch.cat([tmask_0.unsqueeze(0), cmask], dim=0)
            self.b_masks[i] = tmask

        self.collapsed_b_masks = self.b_masks[:, 1, :, :] * 1
        self.b_masks = self.b_masks.to(torch.device("cuda"))
        self.collapsed_b_masks = self.collapsed_b_masks.to(torch.device("cuda"))

        (
            reco,
            z_out,
            mu_tilde,
            a_mu_tilde,
            a_out,
            seg_pred,
            mu,
            logvar,
            a_mu,
            a_logvar,
            input_to_segmentor,
        ) = self.model(imgs, "training")

        if self.charbonnier > 0:
            l1_loss = self.l1_distance(reco, imgs)
            reco_loss = charbonnier_penalty(l1_loss)
        else:
            reco_loss = self.l1_distance(reco, imgs)

        if self.kl_w > 0.0:
            kl_loss = KL_divergence(logvar, mu)
        if self.dice_w > 0.0:
            dice_l = dice_loss(seg_pred[:, 1:, :, :], self.b_masks[:, 1:, :, :])
        if self.regress_w > 0.0:
            regression_loss = self.l1_distance(mu_tilde, z_out)
        if self.focal_w > 0.0:
            self.collapsed_b_masks[self.collapsed_b_masks > 2] = 2
            focal_loss = FocalLoss(gamma=2, alpha=0.25)(
                seg_pred, self.collapsed_b_masks
            )
        else:
            focal_loss = 0.0

        batch_loss = (
            self.reco_w * reco_loss
            + self.kl_w * kl_loss
            + self.dice_w * dice_l
            + self.regress_w * regression_loss
            + self.focal_w * focal_loss
        )

        # Compute dice score for segmentations
        dice_score = torchmetrics.functional.dice(
            seg_pred,
            self.b_masks[:, 1:, :, :].type(torch.uint8),
            num_classes=2,
            ignore_index=0,
        )

        # Plot reconstruction, anatomy channels and segmentations in W&B table
        if self.current_epoch == (self.trainer.max_epochs - 1):
            if batch_idx % 20 == 0:
                anatomy_channels = [
                    [
                        anatomy[0, ...],
                        anatomy[1, ...],
                        anatomy[2, ...],
                        anatomy[3, ...],
                        anatomy[4, ...],
                        anatomy[5, ...],
                        anatomy[6, ...],
                        anatomy[7, ...],
                    ]
                    for anatomy in a_out
                ]
                self.anatomy_train.extend(anatomy_channels)
                general_imgs = [
                    [
                        rec,
                        gt.float(),
                        torch.argmax(pred, 0).float(),
                        img,
                    ]
                    for rec, gt, pred, img in zip(reco, labels, seg_pred, imgs)
                ]
                self.train_table.extend(general_imgs)

        self.log("total_loss", batch_loss, on_step=False, on_epoch=True)
        self.log("reconstruction loss", reco_loss, on_step=False, on_epoch=True)
        self.log("Segmentation loss", dice_l, on_step=False, on_epoch=True)
        self.log("Regression loss", regression_loss, on_step=False, on_epoch=True)
        self.log("Focal loss", focal_loss, on_step=False, on_epoch=True)
        self.log("dice score training", dice_score, on_step=False, on_epoch=True)

        return batch_loss  # Return tensor to call ".backward()" on

    def validation_step(self, batch, batch_idx) -> Optional[Any]:
        imgs, labels, _ = batch
        cmask = labels.unsqueeze(1)
        logical_or = torch.sum(cmask, dim=1)
        tmask_0 = 1 - logical_or
        tmask = torch.cat([tmask_0.unsqueeze(1), cmask], dim=1)

        (reco, a_out, seg_pred, _) = self.model(
            imgs, script_type="validation"
        )  # seg_pred, _

        val_loss = dice_loss(seg_pred[:, 1:, :, :], tmask[:, 1:, :, :])

        # Compute dice score for segmentations
        val_dice = torchmetrics.functional.dice(
            seg_pred,
            tmask[:, 1:, :, :].type(torch.uint8),
            num_classes=2,
            ignore_index=0,
        )
        if self.current_epoch == (self.trainer.max_epochs - 1):
            if batch_idx % 20 == 0:
                anatomy_channels = [
                    [
                        anatomy[0, ...],
                        anatomy[1, ...],
                        anatomy[2, ...],
                        anatomy[3, ...],
                        anatomy[4, ...],
                        anatomy[5, ...],
                        anatomy[6, ...],
                        anatomy[7, ...],
                    ]
                    for anatomy in a_out
                ]
                self.anatomy_val.extend(anatomy_channels)

                general_imgs = [
                    [
                        val_dice,
                        rec,
                        gt.float(),
                        torch.argmax(pred, 0).float(),
                        img,
                    ]
                    for rec, gt, pred, img in zip(reco, labels, seg_pred, imgs)
                ]
                self.val_table.extend(general_imgs)

        self.log("val_dice", val_dice, on_step=False, on_epoch=True)
        self.log("dice loss validation", val_loss, on_step=False, on_epoch=True)

    def on_train_end(self) -> None:
        anatomy_train_wandb = []
        for row in self.anatomy_train:
            anatomy_wand = [
                wandb.Image(row[0]),
                wandb.Image(row[1]),
                wandb.Image(row[2]),
                wandb.Image(row[3]),
                wandb.Image(row[4]),
                wandb.Image(row[5]),
                wandb.Image(row[6]),
                wandb.Image(row[7]),
            ]
            anatomy_train_wandb.append(anatomy_wand)
        self.trainer.logger.log_table(
            key="train anatomy channels",
            columns=self.columns_anatomy,
            data=anatomy_train_wandb,
        )

        train_general_wandb = []
        for row in self.train_table:
            anatomy_wand = [
                wandb.Image(row[0]),
                wandb.Image(row[1]),
                wandb.Image(row[2]),
                wandb.Image(row[3]),
            ]
            train_general_wandb.append(anatomy_wand)
        self.trainer.logger.log_table(
            key="train general table",
            columns=self.columns_table,
            data=train_general_wandb,
        )

        anatomy_val_wandb = []
        for row in self.anatomy_val:
            anatomy_wand = [
                wandb.Image(row[0]),
                wandb.Image(row[1]),
                wandb.Image(row[2]),
                wandb.Image(row[3]),
                wandb.Image(row[4]),
                wandb.Image(row[5]),
                wandb.Image(row[6]),
                wandb.Image(row[7]),
            ]
            anatomy_val_wandb.append(anatomy_wand)
        self.trainer.logger.log_table(
            key="validation anatomy channels",
            columns=self.columns_anatomy,
            data=anatomy_val_wandb,
        )

        val_general_wandb = []
        for row in self.val_table:
            anatomy_wand = [
                row[0],
                wandb.Image(row[1]),
                wandb.Image(row[2]),
                wandb.Image(row[3]),
                wandb.Image(row[4]),
            ]
            val_general_wandb.append(anatomy_wand)
        self.trainer.logger.log_table(
            key="validation general table",
            columns=self.columns_table_val,
            data=val_general_wandb,
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> Optional[Any]:
        imgs, labels, _ = batch
        cmask = labels.unsqueeze(1)
        logical_or = torch.sum(cmask, dim=1)
        tmask_0 = 1 - logical_or
        tmask = torch.cat([tmask_0.unsqueeze(1), cmask], dim=1)
        _, _, seg_pred, _ = self.model(imgs, script_type="testing")
        test_dice = torchmetrics.functional.dice(
            seg_pred,
            tmask[:, 1:, :, :].type(torch.uint8),
            zero_division=1,
            num_classes=2,
            ignore_index=0,
        )
        recall_score = torchmetrics.functional.classification.binary_recall(
            torch.argmax(seg_pred, 1),
            tmask[:, 1:, :, :].type(torch.uint8).squeeze(1),
            threshold=0.5,
            multidim_average="global",
            ignore_index=0,
            validate_args=True,
        )
        accuracy_score = torchmetrics.functional.classification.binary_accuracy(
            torch.argmax(seg_pred, 1),
            tmask[:, 1:, :, :].type(torch.uint8).squeeze(1),
            threshold=0.5,
            multidim_average="global",
            ignore_index=None,
            validate_args=True,
        )

        self.log_dict(
            {
                f"{dataloader_idx}_test_dice": test_dice,
                f"{dataloader_idx}_recall": recall_score,
                f"{dataloader_idx}_accuracy": accuracy_score,
            }
        )

    def on_test_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        imgs, labels, centre = batch
        cmask = labels.unsqueeze(1)
        logical_or = torch.sum(cmask, dim=1)
        tmask_0 = 1 - logical_or
        tmask = torch.cat([tmask_0.unsqueeze(1), cmask], dim=1)
        reco, a_out, seg_pred, _ = self.model(
            imgs, script_type="testing"
        )  # seg_pred, _

        anatomy_channels = [
            [
                anatomy[0, ...],
                anatomy[1, ...],
                anatomy[2, ...],
                anatomy[3, ...],
                anatomy[4, ...],
                anatomy[5, ...],
                anatomy[6, ...],
                anatomy[7, ...],
            ]
            for anatomy in a_out
        ]

        test_dice = torchmetrics.functional.dice(
            seg_pred,
            tmask[:, 1:, :, :].type(torch.uint8),
            zero_division=1,
            num_classes=2,
            ignore_index=0,
        )
        recall_score = torchmetrics.functional.classification.binary_recall(
            torch.argmax(seg_pred, 1),
            tmask[:, 1:, :, :].type(torch.uint8).squeeze(1),
            threshold=0.5,
            multidim_average="global",
            ignore_index=0,
            validate_args=True,
        )
        accuracy_score = torchmetrics.functional.classification.binary_accuracy(
            torch.argmax(seg_pred, 1),
            tmask[:, 1:, :, :].type(torch.uint8).squeeze(1),
            threshold=0.5,
            multidim_average="global",
            ignore_index=None,
            validate_args=True,
        )
        general_imgs = [
            [
                centre.to(torch.float),
                test_dice,
                recall_score,
                accuracy_score,
                rec,
                gt.float(),
                torch.argmax(pred, 0).float(),
                img,
            ]
            for rec, gt, pred, img in zip(reco, labels, seg_pred, imgs)
        ]

        if dataloader_idx == 0:
            self.anatomy_in_test.extend(anatomy_channels)
            self.in_test_table.extend(general_imgs)
        # else:
        #     self.anatomy_out_test.extend(anatomy_channels)
        #     self.out_test_table.extend(general_imgs)

    def on_test_end(self) -> None:
        anatomy_in_test_wandb = []
        for row in self.anatomy_in_test:
            anatomy_wand = [
                wandb.Image(row[0]),
                wandb.Image(row[1]),
                wandb.Image(row[2]),
                wandb.Image(row[3]),
                wandb.Image(row[4]),
                wandb.Image(row[5]),
                wandb.Image(row[6]),
                wandb.Image(row[7]),
            ]
            anatomy_in_test_wandb.append(anatomy_wand)
        self.trainer.logger.log_table(
            key="in distribution test anatomy channels",
            columns=self.columns_anatomy,
            data=anatomy_in_test_wandb,
        )

        in_test_table_wandb = []
        for row in self.in_test_table:
            anatomy_wand = [
                row[0],
                row[1],
                row[2],
                row[3],
                wandb.Image(row[4]),
                wandb.Image(row[5]),
                wandb.Image(row[6]),
                wandb.Image(row[7]),
            ]
            in_test_table_wandb.append(anatomy_wand)
        self.trainer.logger.log_table(
            key="in distribution test table",
            columns=self.columns_table_test,
            data=in_test_table_wandb,
        )

        # anatomy_out_test_wandb = []
        # for row in self.anatomy_out_test:
        #     anatomy_wand = [
        #         wandb.Image(row[0]),
        #         wandb.Image(row[1]),
        #         wandb.Image(row[2]),
        #         wandb.Image(row[3]),
        #         wandb.Image(row[4]),
        #         wandb.Image(row[5]),
        #         wandb.Image(row[6]),
        #         wandb.Image(row[7]),
        #     ]
        #     anatomy_out_test_wandb.append(anatomy_wand)
        # self.trainer.logger.log_table(
        #     key="out distribution test anatomy channels",
        #     columns=self.columns_anatomy,
        #     data=anatomy_out_test_wandb,
        # )

        # out_test_table_wandb = []
        # for row in self.out_test_table:
        #     anatomy_wand = [
        #         row[0],
        #         row[1],
        #         row[2],
        #         row[3],
        #         wandb.Image(row[4]),
        #         wandb.Image(row[5]),
        #         wandb.Image(row[6]),
        #         wandb.Image(row[7]),
        #     ]
        #     out_test_table_wandb.append(anatomy_wand)
        # self.trainer.logger.log_table(
        #     key="out distribution test table",
        #     columns=self.columns_table_test,
        #     data=out_test_table_wandb,
        # )
