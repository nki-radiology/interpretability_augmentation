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
import kornia as K


class SDNetAnatomyDisLight(pl.LightningModule):
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
        self.num_classes = 5
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
        self.columns_anatomy_train = [
            "channel_1",
            "channel_2",
            "channel_3",
            "channel_4",
            "channel_5",
        ]
        self.centres_channels = []
        self.columns_centres_channels = ["channel_6", "channel_7", "channel_8"]
        self.train_table = []
        self.columns_train_table = [
            "centre",
            "predicted centre",
            "reconstruction",
            "ground_truth",
            "prediction",
            "image",
        ]
        self.anatomy_val = []
        self.columns_anatomy_val = [
            "channel_1",
            "channel_2",
            "channel_3",
            "channel_4",
            "channel_5",
            "channel_6",
            "channel_7",
            "channel_8",
        ]
        self.val_table = []
        self.columns_val_table = [
            "dice",
            "reconstruction",
            "ground truth",
            "prediction",
            "image",
        ]
        self.anatomy_in_test = []
        self.anatomy_out_test = []
        self.in_test_table = []
        self.out_test_table = []
        self.columns_test_table = [
            "centre",
            "dice",
            "reconstruction",
            "ground truth",
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
        imgs, labels, centres = batch

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
            cls,
            cls_adv,
        ) = self.model(imgs, script_type="training")

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
        # cls = torch.argmax(cls, dim=1)
        classifier_loss = self.cross_entropy(cls, centres)
        # 100 is an arbitrary value, might be too high (TODO: check if it makes sense or if there should be an upper bound)
        uniform_dist = torch.ones(cls_adv.shape[0], 5) * (1 / 5)
        uniform_dist = uniform_dist.type_as(cls_adv)
        adv_loss = self.cross_entropy(cls_adv, uniform_dist)

        batch_loss = (
            self.reco_w * reco_loss
            + self.kl_w * kl_loss
            + self.dice_w * dice_l
            + self.regress_w * regression_loss
            + self.focal_w * focal_loss
            + classifier_loss
            + adv_loss
        )

        # Compute dice score for segmentations
        dice_score = torchmetrics.functional.dice(
            seg_pred,
            self.b_masks[:, 1:, :, :].type(torch.uint8),
            num_classes=2,
            ignore_index=0,
        )

        # Plot reconstruction, anatomy channels and segmentations in W&B table
        if self.current_epoch % 30:
            if batch_idx == 5:
                anatomy_channels = [
                    [
                        anatomy[3, ...],
                        anatomy[4, ...],
                        anatomy[5, ...],
                        anatomy[6, ...],
                        anatomy[7, ...],
                    ]
                    for anatomy in a_out
                ]
                self.anatomy_train.extend(anatomy_channels)
                content_channels = [
                    [
                        anatomy[0, ...],
                        anatomy[1, ...],
                        anatomy[2, ...],
                    ]
                    for anatomy in a_out
                ]
                self.centres_channels.extend(content_channels)
                cls = torch.argmax(cls, 1)
                general_imgs = [
                    [
                        centre.float(),
                        cl.float(),
                        rec,
                        gt.float(),
                        torch.argmax(pred, 0).float(),
                        img,
                    ]
                    for centre, cl, rec, gt, pred, img in zip(
                        centres,
                        cls,
                        reco,
                        labels,
                        seg_pred,
                        imgs,
                    )
                ]
                self.train_table.extend(general_imgs)

        self.log("total_loss", batch_loss, on_step=False, on_epoch=True)
        self.log("reconstruction loss", reco_loss, on_step=False, on_epoch=True)
        self.log("Segmentation loss", dice_l, on_step=False, on_epoch=True)
        self.log("Regression loss", regression_loss, on_step=False, on_epoch=True)
        self.log("Focal loss", focal_loss, on_step=False, on_epoch=True)
        self.log(
            "Centres classification loss", classifier_loss, on_step=False, on_epoch=True
        )
        self.log("Adversarial loss", adv_loss, on_step=False, on_epoch=True)
        self.log("dice score training", dice_score, on_step=False, on_epoch=True)

        return batch_loss  # Return tensor to call ".backward()" on

    def validation_step(self, batch, batch_idx) -> Optional[Any]:
        with torch.enable_grad():
            imgs, labels, _ = batch
            cmask = labels.unsqueeze(1)
            logical_or = torch.sum(cmask, dim=1)
            tmask_0 = 1 - logical_or
            tmask = torch.cat([tmask_0.unsqueeze(1), cmask], dim=1)

            reco, a_out, seg_pred = self.model(imgs, script_type="validation")

            val_loss = dice_loss(seg_pred[:, 1:, :, :], tmask[:, 1:, :, :])
            if self.charbonnier > 0:
                l1_loss = self.l1_distance(reco, imgs)
                val_reco_loss = charbonnier_penalty(l1_loss)
            else:
                val_reco_loss = self.l1_distance(reco, imgs)

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
                        for rec, gt, pred, img in zip(
                            reco,
                            labels,
                            seg_pred,
                            imgs,
                        )
                    ]
                    self.val_table.extend(general_imgs)

            self.log("val_dice", val_dice, on_step=False, on_epoch=True)
            self.log("dice loss validation", val_loss, on_step=False, on_epoch=True)
            self.log(
                "reconstruction loss validation",
                val_reco_loss,
                on_step=False,
                on_epoch=True,
            )

    def on_train_end(self) -> None:
        anatomy_wand_final = []
        for row in self.anatomy_train:
            anatomy_wand = [
                wandb.Image(row[0]),
                wandb.Image(row[1]),
                wandb.Image(row[2]),
                wandb.Image(row[3]),
                wandb.Image(row[4]),
            ]
            anatomy_wand_final.append(anatomy_wand)

        centres_channels = []
        for row in self.centres_channels:
            anatomy_wand = [
                wandb.Image(row[0]),
                wandb.Image(row[1]),
                wandb.Image(row[2]),
            ]
            centres_channels.append(anatomy_wand)

        train_table_wandb = []
        for row in self.train_table:
            anatomy_wand = [
                row[0],
                row[1],
                wandb.Image(row[2]),
                wandb.Image(row[3]),
                wandb.Image(row[4]),
                wandb.Image(row[5]),
            ]
            train_table_wandb.append(anatomy_wand)

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

        val_table_wandb = []
        for row in self.val_table:
            anatomy_wand = [
                row[0],
                wandb.Image(row[1]),
                wandb.Image(row[2]),
                wandb.Image(row[3]),
                wandb.Image(row[4]),
            ]
            val_table_wandb.append(anatomy_wand)

        self.trainer.logger.log_table(
            key="train anatomy channels",
            columns=self.columns_anatomy_train,
            data=anatomy_wand_final,
        )
        self.trainer.logger.log_table(
            key="centre content channels",
            columns=self.columns_centres_channels,
            data=centres_channels,
        )
        self.trainer.logger.log_table(
            key="train general table",
            columns=self.columns_train_table,
            data=train_table_wandb,
        )
        self.trainer.logger.log_table(
            key="validation anatomy channels",
            columns=self.columns_anatomy_val,
            data=anatomy_val_wandb,
        )
        self.trainer.logger.log_table(
            key="validation general table",
            columns=self.columns_val_table,
            data=val_table_wandb,
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

        general_imgs = [
            [
                centre.to(torch.float),
                test_dice,
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
        else:
            self.anatomy_out_test.extend(anatomy_channels)
            self.out_test_table.extend(general_imgs)

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

        in_test_wandb = []
        for row in self.in_test_table:
            anatomy_wand = [
                row[0],
                row[1],
                wandb.Image(row[2]),
                wandb.Image(row[3]),
                wandb.Image(row[4]),
                wandb.Image(row[5]),
            ]
            in_test_wandb.append(anatomy_wand)

        anatomy_out_test_wandb = []
        for row in self.anatomy_out_test:
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
            anatomy_out_test_wandb.append(anatomy_wand)

        out_test_wandb = []
        for row in self.out_test_table:
            anatomy_wand = [
                row[0],
                row[1],
                wandb.Image(row[2]),
                wandb.Image(row[3]),
                wandb.Image(row[4]),
                wandb.Image(row[5]),
            ]
            out_test_wandb.append(anatomy_wand)

        self.trainer.logger.log_table(
            key="in distribution test anatomy channels",
            columns=self.columns_anatomy_val,
            data=anatomy_in_test_wandb,
        )
        self.trainer.logger.log_table(
            key="in distribution test table",
            columns=self.columns_test_table,
            data=in_test_wandb,
        )

        self.trainer.logger.log_table(
            key="out distribution test anatomy channels",
            columns=self.columns_anatomy_val,
            data=anatomy_out_test_wandb,
        )
        self.trainer.logger.log_table(
            key="out distribution test table",
            columns=self.columns_test_table,
            data=out_test_wandb,
        )
