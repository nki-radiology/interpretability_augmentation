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
from NewSDNet.models.GANSDNet import GANSDNet
from torchvision.models import resnet50

"""NOTE to self tomorrow: the problem is probably that you tried to toggle multiple optimizers at the same time and this is not allowed.
                          Idea: group all sdnet module in one class and assign to it one optimizer; this should work, you have all optimizers in one class. 
"""


class GANSDNetLightning(pl.LightningModule):
    def __init__(
        self,
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
        z_lenght: int,
        height: int,
        width: int,
        ndf: int,
        anatomy_out_channels: int,
        classifier_ckpt: str,
        norm,
        upsample,
        num_mask_channels,
    ):
        super().__init__()
        # Deactivate automatic optimization for adversarial training

        # self.save_hyperparameters()
        # self.automatic_optimization = False

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
        self.classifier_ckpt = classifier_ckpt
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
            "predicted centre by adv classifier",
            "predictied centre by anatomy classifier",
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
        self.z_lenght = z_lenght
        self.h = height
        self.w = width
        self.ndf = ndf
        self.anatomy_out_channels = anatomy_out_channels
        self.norm = norm
        self.upsample = upsample
        self.num_mask_channels = num_mask_channels

        # Initialize the network modules and their weights
        # self.m_encoder = GANSDNet.MEncoder(self.z_lenght)
        # self.a_encoder = GANSDNet.AEncoder(
        #     self.h,
        #     self.w,
        #     self.ndf,
        #     self.anatomy_out_channels,
        #     self.norm,
        #     self.upsample,
        # )
        # self.segmentor = GANSDNet.Segmentor(8, 1)
        # self.decoder = GANSDNet.Decoder(
        #     self.anatomy_out_channels, self.z_lenght, self.num_mask_channels
        # )
        self.sdnet = GANSDNet.SDNet(
            width=self.w,
            height=self.h,
            num_classes=1,
            ndf=self.ndf,
            z_length=8,
            norm=self.norm,
            upsample=self.upsample,
            anatomy_out_channels=8,
            num_mask_channels=self.num_mask_channels,
        )

        initialize_weights(self.sdnet, self.weight_init)

        self.classifier = resnet50()
        linear_size = list(self.classifier.children())[-1].in_features
        self.classifier.fc = nn.Linear(linear_size, self.num_classes)
        self.classifier.load_state_dict(torch.load(self.classifier_ckpt))
        # self.channel_classifier = resnet50()
        # self.channel_classifier.conv1 = nn.Conv2d(
        #     1, 64, kernel_size=7, stride=2, padding=3, bias=False
        # )
        # linear_size = list(self.channel_classifier.children())[-1].in_features
        # self.channel_classifier.fc = nn.Linear(
        #     linear_size, 6
        # )  # 6 like the number of channels where we want content to be located

    def forward(self, imgs, script_type="validation"):
        return self.sdnet(imgs)

    def configure_optimizers(self) -> Any:
        # opt_sdnet = optim.AdamW(self.sdnet.parameters(), lr=self.lr)
        # opt_classifier = optim.AdamW(self.classifier.parameters(), lr=self.lr)
        # self.parameters

        return optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        imgs, labels, centres = batch

        for i in range(labels.shape[0]):
            cmask = labels[i].unsqueeze(0)
            logical_or = torch.sum(cmask, dim=0)
            tmask_0 = 1 - logical_or
            tmask = torch.cat([tmask_0.unsqueeze(0), cmask], dim=0)
            self.b_masks[i] = tmask

        self.collapsed_b_masks = self.b_masks[:, 1, :, :] * 1
        self.b_masks = self.b_masks.to(torch.device("cuda"))
        self.collapsed_b_masks = self.collapsed_b_masks.to(torch.device("cuda"))

        # # Get optimizers
        # (
        #     opt_sdnet,
        #     opt_classifier,
        # ) = self.optimizers()

        ####  ####
        (
            reco,
            z_out,
            mu_out_tilde,
            a_out_tilde,
            a_out,
            seg_pred,
            mu_out,
            logvar_out,
        ) = self.sdnet(imgs, script_type="training")
        ####Optimize discriminator ####
        # torch.autograd.set_detect_anomaly(True)
        # a_out_new = a_out.detach()
        # real_predictions = self.classifier(a_out[:, :3, :, :].detach())
        # confusing_predictions = self.classifier(a_out[:, 3:6, :, :].detach())
        # uniform_dist = torch.ones(real_predictions.shape[0], 5) * (1 / 5)
        # uniform_dist = uniform_dist.type_as(real_predictions)
        # real_loss = self.cross_entropy(
        #     real_predictions, uniform_dist
        # )  # uniform_dist and centres are inverted w.r.t the generator training step to create opposite objective (TODO: check with Wilson that it makes sense)
        # confusing_loss = self.cross_entropy(confusing_predictions, centres)
        # total_classifier_loss = (real_loss + confusing_loss) / 2
        # total_classifier_loss.requires_grad_()
        # opt_classifier.zero_grad()
        # self.manual_backward(total_classifier_loss)
        # opt_classifier.step()
        # logging
        # self.log(
        #     "Discriminator loss",
        #     total_classifier_loss,
        #     on_step=False,
        #     on_epoch=True,
        # )

        # Compute losses

        adv_predictions = self.classifier(a_out[:, :3, :, :])
        anatomy_predictions = self.classifier(a_out[:, 3:6, :, :])

        # Compute predictions for the channels classfier
        # anatomy_content = a_out[:, :6, :, :].type_as(a_out)
        # indices_content = torch.randperm(anatomy_content.size()[1])
        # anatomy_content_shuffled = anatomy_content[:, indices_content, :, :].type_as(
        #     a_out
        # )
        # labels_anatomy_content = (
        #     torch.arange(0, anatomy_content.size()[1], step=1)
        #     .repeat(anatomy_content.size()[0], 1)
        #     .type_as(centres)
        # )
        # labels_anatomy_content_shuffled = labels_anatomy_content[
        #     :, indices_content
        # ].type_as(centres)
        # loss_per_channel_classifier = 0.0
        # for i in range(0, anatomy_content.size()[1]):
        #     content_predictions = self.channel_classifier(
        #         anatomy_content_shuffled[:, i, :, :].unsqueeze(1)
        #     )
        #     loss_channel_classifier = self.cross_entropy(
        #         content_predictions, labels_anatomy_content_shuffled[:, i].squeeze()
        #     )
        #     loss_per_channel_classifier += loss_channel_classifier
        # loss_channel_classifier_total = (
        #     loss_per_channel_classifier / anatomy_content.size()[1]
        # )
        if self.charbonnier > 0:
            l1_loss = self.l1_distance(reco, imgs)
            reco_loss = charbonnier_penalty(l1_loss)
        else:
            reco_loss = self.l1_distance(reco, imgs)
        if self.kl_w > 0.0:
            kl_loss = KL_divergence(logvar_out, mu_out)
        if self.dice_w > 0.0:
            dice_l = dice_loss(seg_pred[:, 1:, :, :], self.b_masks[:, 1:, :, :])
        if self.regress_w > 0.0:
            regression_loss = self.l1_distance(mu_out_tilde, z_out)
        # if self.focal_w > 0.0:
        #     self.collapsed_b_masks_gpu[self.collapsed_b_masks > 2] = 2
        #     focal_loss = FocalLoss(gamma=2, alpha=0.25)(
        #         seg_pred, self.collapsed_b_masks
        #     )
        # else:
        #     focal_loss = 0.0
        uniform_dist = torch.ones(anatomy_predictions.shape[0], 5) * (1 / 5)
        uniform_dist = uniform_dist.type_as(anatomy_predictions)
        adversarial_loss = self.cross_entropy(
            adv_predictions, centres
        )  # we want the generator to make the first 3 channels more related to the centres
        second_confusing_loss = self.cross_entropy(
            anatomy_predictions, uniform_dist
        )  # we want the generator to work towards making channels 3-6 anatomy related

        total_adv_loss = (adversarial_loss + second_confusing_loss) / 2
        # total_adv_loss = adversarial_loss  # dummy assignment
        total_first_step_loss = (
            self.reco_w * reco_loss
            + self.kl_w * kl_loss
            + self.dice_w * dice_l
            + self.regress_w * regression_loss
            + total_adv_loss
            # + loss_channel_classifier_total
            # + self.focal_w * focal_loss
        )
        # opt_sdnet.zero_grad()
        # self.manual_backward(total_first_step_loss)
        # opt_sdnet.step()

        # logging
        self.log("reconstruction loss", reco_loss, on_step=False, on_epoch=True)
        self.log("Segmentation loss", dice_l, on_step=False, on_epoch=True)
        self.log("Regression loss", regression_loss, on_step=False, on_epoch=True)
        # self.log("Focal loss", focal_loss, on_step=False, on_epoch=True)
        self.log("KL", kl_loss, on_step=False, on_epoch=True)
        self.log(
            "Centres classification loss in 'generator_training'",
            total_adv_loss,
            on_step=False,
            on_epoch=True,
        )
        # self.log(
        #     "Channels content classifier",
        #     loss_channel_classifier_total,
        #     on_step=False,
        #     on_epoch=True,
        # )

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
                adv_predictions = torch.argmax(adv_predictions, 1)
                anatomy_predictions = torch.argmax(anatomy_predictions, 1)
                general_imgs = [
                    [
                        centre.float(),
                        adv_pred.float(),
                        an_pred.float(),
                        rec,
                        gt.float(),
                        torch.argmax(pred, 0).float(),
                        img,
                    ]
                    for centre, adv_pred, an_pred, rec, gt, pred, img in zip(
                        centres,
                        adv_predictions,
                        anatomy_predictions,
                        reco,
                        labels,
                        seg_pred,
                        imgs,
                    )
                ]
                self.train_table.extend(general_imgs)

        self.log("dice score training", dice_score, on_step=False, on_epoch=True)
        return total_first_step_loss

    def validation_step(self, batch, batch_idx) -> Optional[Any]:
        imgs, labels, _ = batch
        cmask = labels.unsqueeze(1)
        logical_or = torch.sum(cmask, dim=1)
        tmask_0 = 1 - logical_or
        tmask = torch.cat([tmask_0.unsqueeze(1), cmask], dim=1)

        reco, a_out, seg_pred = self.sdnet(imgs, script_type="validation")

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
                row[2],
                wandb.Image(row[3]),
                wandb.Image(row[4]),
                wandb.Image(row[5]),
                wandb.Image(row[6]),
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
        _, _, seg_pred = self.sdnet(imgs, script_type="testing")

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
        reco, a_out, seg_pred = self.sdnet(imgs, script_type="testing")

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
