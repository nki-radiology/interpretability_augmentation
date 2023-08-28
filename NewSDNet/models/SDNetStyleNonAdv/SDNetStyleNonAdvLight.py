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


class SDNetStyleNonAdvLightning(pl.LightningModule):
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
        self.cross_entropy = nn.CrossEntropyLoss()
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
        imgs, labels, centre = batch

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
            mu_out_tilde_common_style,
            a_out_tilde,
            a_out,
            seg_pred,
            mu_out,
            logvar_out,
            a_mu_out,
            a_logvar_out,
            z_out_domain,
            mu_out_domain,
            logvar_out_domain,
            mu_out_tilde_domain_style,
            cls,
        ) = self.model(imgs, "training")

        if self.charbonnier > 0:
            l1_loss = self.l1_distance(reco, imgs)
            reco_loss = charbonnier_penalty(l1_loss)
        else:
            reco_loss = self.l1_distance(reco, imgs)

        if self.kl_w > 0.0:
            kl_loss_style = KL_divergence(logvar_out, mu_out)
            kl_loss_domain = KL_divergence(logvar_out_domain, mu_out_domain)
            hsic_loss = HSIC_lossfunc(z_out, z_out_domain)
            kl_loss = kl_loss_style + kl_loss_domain + hsic_loss
        if self.dice_w > 0.0:
            dice_l = dice_loss(seg_pred[:, 1:, :, :], self.b_masks[:, 1:, :, :])
        if self.regress_w > 0.0:
            regression_loss_style = self.l1_distance(mu_out_tilde_common_style, z_out)
            regression_loss_domain = self.l1_distance(
                mu_out_tilde_domain_style, z_out_domain
            )
            regression_loss = 0.5 * regression_loss_style + 0.5 * regression_loss_domain
        if self.focal_w > 0.0:
            self.collapsed_b_masks[self.collapsed_b_masks > 2] = 2
            focal_loss = FocalLoss(gamma=2, alpha=0.25)(
                seg_pred, self.collapsed_b_masks
            )
        else:
            focal_loss = 0.0

        classification_loss = self.cross_entropy(cls, centre)

        batch_loss = (
            self.reco_w * reco_loss
            + self.kl_w * kl_loss
            + self.dice_w * dice_l
            + self.regress_w * regression_loss
            + self.focal_w * focal_loss
            + classification_loss
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
                        wandb.Image(anatomy[0, ...]),
                        wandb.Image(anatomy[1, ...]),
                        wandb.Image(anatomy[2, ...]),
                        wandb.Image(anatomy[3, ...]),
                        wandb.Image(anatomy[4, ...]),
                        wandb.Image(anatomy[5, ...]),
                        wandb.Image(anatomy[6, ...]),
                        wandb.Image(anatomy[7, ...]),
                    ]
                    for anatomy in a_out
                ]
                self.anatomy_train.extend(anatomy_channels)
                general_imgs = [
                    [
                        wandb.Image(rec),
                        wandb.Image(gt.float()),
                        wandb.Image(torch.argmax(pred, 0).float()),
                        wandb.Image(img),
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
        self.log(
            "classification loss", classification_loss, on_step=False, on_epoch=True
        )

        return batch_loss  # Return tensor to call ".backward()" on

    def validation_step(self, batch, batch_idx) -> Optional[Any]:
        imgs, labels, _ = batch
        cmask = labels.unsqueeze(1)
        logical_or = torch.sum(cmask, dim=1)
        tmask_0 = 1 - logical_or
        tmask = torch.cat([tmask_0.unsqueeze(1), cmask], dim=1)

        reco, a_out, seg_pred = self.model(imgs, script_type="validation")

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
                        wandb.Image(anatomy[0, ...]),
                        wandb.Image(anatomy[1, ...]),
                        wandb.Image(anatomy[2, ...]),
                        wandb.Image(anatomy[3, ...]),
                        wandb.Image(anatomy[4, ...]),
                        wandb.Image(anatomy[5, ...]),
                        wandb.Image(anatomy[6, ...]),
                        wandb.Image(anatomy[7, ...]),
                    ]
                    for anatomy in a_out
                ]
                self.anatomy_val.extend(anatomy_channels)

                general_imgs = [
                    [
                        val_dice,
                        wandb.Image(rec),
                        wandb.Image(gt.float()),
                        wandb.Image(torch.argmax(pred, 0).float()),
                        wandb.Image(img),
                    ]
                    for rec, gt, pred, img in zip(reco, labels, seg_pred, imgs)
                ]
                self.val_table.extend(general_imgs)

        self.log("val_dice", val_dice, on_step=False, on_epoch=True)
        self.log("dice loss validation", val_loss, on_step=False, on_epoch=True)

    def on_train_end(self) -> None:
        self.trainer.logger.log_table(
            key="train anatomy channels",
            columns=self.columns_anatomy,
            data=self.anatomy_train,
        )
        self.trainer.logger.log_table(
            key="train general table",
            columns=self.columns_table,
            data=self.train_table,
        )
        self.trainer.logger.log_table(
            key="validation anatomy channels",
            columns=self.columns_anatomy,
            data=self.anatomy_val,
        )
        self.trainer.logger.log_table(
            key="validation general table",
            columns=self.columns_table_val,
            data=self.val_table,
        )

    def test_step(self, batch, batch_idx, dataloader_idx) -> Optional[Any]:
        imgs, labels, _ = batch
        cmask = labels.unsqueeze(1)
        logical_or = torch.sum(cmask, dim=1)
        tmask_0 = 1 - logical_or
        tmask = torch.cat([tmask_0.unsqueeze(1), cmask], dim=1)
        _, _, seg_pred = self.model(imgs, script_type="testing")
        test_dice = torchmetrics.functional.dice(
            seg_pred,
            tmask[:, 1:, :, :].type(torch.uint8),
            zero_division=1,
            num_classes=2,
            ignore_index=0,
        )

        self.log(f"{dataloader_idx}_test_dice", test_dice, on_step=False, on_epoch=True)

    def on_test_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        imgs, labels, centre = batch
        cmask = labels.unsqueeze(1)
        logical_or = torch.sum(cmask, dim=1)
        tmask_0 = 1 - logical_or
        tmask = torch.cat([tmask_0.unsqueeze(1), cmask], dim=1)
        reco, a_out, seg_pred = self.model(imgs, script_type="testing")

        anatomy_channels = [
            [
                wandb.Image(anatomy[0, ...]),
                wandb.Image(anatomy[1, ...]),
                wandb.Image(anatomy[2, ...]),
                wandb.Image(anatomy[3, ...]),
                wandb.Image(anatomy[4, ...]),
                wandb.Image(anatomy[5, ...]),
                wandb.Image(anatomy[6, ...]),
                wandb.Image(anatomy[7, ...]),
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

        general_imgs = [
            [
                centre.to(torch.float),
                test_dice,
                wandb.Image(rec),
                wandb.Image(gt.float()),
                wandb.Image(torch.argmax(pred, 0).float()),
                wandb.Image(img),
            ]
            for rec, gt, pred, img in zip(reco, labels, seg_pred, imgs)
        ]

        if dataloader_idx == 0:
            self.anatomy_in_test.extend(anatomy_channels)
            self.in_test_table.extend(general_imgs)
        else:
            self.anatomy_out_test.extend(anatomy_channels)
            self.out_test_table.extend(general_imgs)

    def on_test_end(self) -> None:
        self.trainer.logger.log_table(
            key="in distribution test anatomy channels",
            columns=self.columns_anatomy,
            data=self.anatomy_in_test,
        )
        self.trainer.logger.log_table(
            key="in distribution test table",
            columns=self.columns_table_test,
            data=self.in_test_table,
        )

        self.trainer.logger.log_table(
            key="out distribution test anatomy channels",
            columns=self.columns_anatomy,
            data=self.anatomy_out_test,
        )
        self.trainer.logger.log_table(
            key="out distribution test table",
            columns=self.columns_table_test,
            data=self.out_test_table,
        )
