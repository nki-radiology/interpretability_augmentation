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
import kornia as K
from monai.visualize import CAM
from monai.visualize.gradient_based import GuidedBackpropGrad


# NOTE assume model output of shape (batch,2classes,H,W) with logits output
# --> for cross entropy loss


class ModifiedUNetLightning(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        # classifier: torch.nn.Module,
        # classifier_ckpt_path: str,
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
        # self.classifier = classifier
        # self.classifier_ckpt_path = classifier_ckpt_path
        self.batch_size = batch_size
        self.num_classes = 5
        # Create loss module
        self.loss_segmentation = nn.CrossEntropyLoss()
        # self.loss_reconstruction = nn.L1Loss()
        self.lr = lr
        self.img_logger = img_logger
        self.save_path = save_path
        self.max_dice_train = -1
        self.train_table: list[list[Any]] = []
        self.columns_train: list[str] = [
            "centre",
            "dice score",
            "image",
            "saliency map",
            "gradcam prediction",
            # "opposite saliency map",
            "ground truth",
            "prediction",
        ]
        self.columns_val: list[str] = [
            "centre",
            "dice score",
            "image",
            # "opposite saliency map",
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
        # linear_size = list(self.classifier.children())[-1].in_features
        # self.classifier.fc = nn.Linear(linear_size, self.num_classes)
        # self.classifier.load_state_dict(torch.load(self.classifier_ckpt_path))
        # self.cam = CAM(
        #     nn_module=self.classifier, target_layers="layer4.2.relu", fc_layers="fc"
        # )
        # self.guided_backprop = GuidedBackpropGrad(model=self.classifier)

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
        imgs, labels, _, bin_saliency_maps = batch
        imgs_gray = K.color.rgb_to_grayscale(imgs)
        imgs_sobel: torch.Tensor = K.filters.sobel(imgs_gray)
        sobel_to_rgb: torch.Tensor = imgs_sobel.repeat(1, 3, 1, 1)

        ### Using cam for interpretability ###
        # saliency_maps_batch: list[torch.Tensor] = []
        # for img_to_rgb in sobel_to_rgb:
        #     saliency_map = self.cam(img_to_rgb.unsqueeze(0))
        #     saliency_maps_batch.append(saliency_map)
        # saliency_maps = torch.cat(saliency_maps_batch, dim=0)
        ### Using guided backprop for interpretability ###
        # saliency_maps_batch: list[torch.Tensor] = []
        # for img_to_rgb in imgs_to_rgb:
        #     saliency_map = self.guided_backprop(img_to_rgb.unsqueeze(0))
        #     saliency_maps_batch.append(saliency_map)
        # saliency_maps = torch.cat(saliency_maps_batch, dim=0)

        # bin_saliency_maps = torch.where(saliency_maps > 0.5, 1, 0)
        bin_saliency_maps_1_ch = bin_saliency_maps[
            :, 0, :, :
        ]  # it's 8 channel for the sdnet, not optimal doing it now but still
        # opposite_saliency_maps = torch.where(
        #     bin_saliency_maps == 0.0, 1, 0
        # )  # no need when using cam
        # imgs_to_reconstruct = imgs * opposite_saliency_maps
        # imgs_to_reconstruct = imgs * bin_saliency_maps_3_ch

        segmentation_preds, gradcam_preds = self.model(
            imgs, script_type="training"
        )  # features shape = [8, 512, 32, 32]
        loss_seg = self.loss_segmentation(segmentation_preds, labels)
        loss_gradcam_seg = self.loss_segmentation(gradcam_preds, bin_saliency_maps_1_ch)

        loss = 0.5 * loss_seg + 0.5 * loss_gradcam_seg
        # compute the dice score
        dice_score = torchmetrics.functional.dice(
            segmentation_preds, labels, zero_division=1, num_classes=2, ignore_index=0
        )

        self.log("train_dice", dice_score, on_step=False, on_epoch=True)
        self.log("total_train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_segmentation_loss", loss_seg, on_step=False, on_epoch=True)
        self.log(
            "train_recontruction_loss", loss_gradcam_seg, on_step=False, on_epoch=True
        )
        self.log("RAM", psutil.virtual_memory().percent, on_step=True)
        return {
            "loss": loss,
            "segmentation_preds": segmentation_preds,
            "gradcam_preds": gradcam_preds,
            "bin_saliency_maps": bin_saliency_maps_1_ch,
            # "opposite_saliency_maps": opposite_saliency_maps,
        }  # Return tensor to call ".backward" on

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        imgs, labels, centre, _ = batch
        segmentation_preds = outputs["segmentation_preds"]
        gradcam_preds = outputs["gradcam_preds"]
        # imgs_to_reconstruct = outputs["imgs_to_reconstruct"]
        bin_saliency_maps = outputs["bin_saliency_maps"]
        # opposite_saliency_maps = outputs["opposite_saliency_maps"]

        dice = torchmetrics.functional.dice(
            segmentation_preds,
            labels,
            zero_division=1,
            num_classes=2,
            ignore_index=0,
        )
        segmentation_preds_to_plot = torch.argmax(segmentation_preds, 1)
        gradcam_preds_to_plot = torch.argmax(gradcam_preds, 1)
        if self.current_epoch == (self.trainer.max_epochs - 1):
            if batch_idx % 10 == 0:
                images = [
                    [
                        centre.to(torch.float),
                        dice,
                        img,
                        bin_saliency_map.float(),
                        gradcam_pred.float(),
                        # wandb.Image(opposite_saliency_map.float()),
                        label.float(),
                        segmentation_pred.float(),
                    ]
                    for img, bin_saliency_map, gradcam_pred, label, segmentation_pred in zip(
                        imgs,
                        bin_saliency_maps,
                        gradcam_preds_to_plot,
                        # opposite_saliency_maps,
                        labels,
                        segmentation_preds_to_plot,
                    )
                ]
                self.train_table.extend(images)

    def validation_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        with torch.enable_grad():
            imgs, labels, _ = batch
            # imgs_gray = K.color.rgb_to_grayscale(imgs)
            # imgs_sobel: torch.Tensor = K.filters.sobel(imgs_gray)
            # imgs_to_rgb: torch.Tensor = imgs_sobel.repeat(1, 3, 1, 1)
            # imgs_kornia = K.color.grayscale_to_rgb(imgs_sobel)

            ### Using cam for interpretability ###
            # saliency_maps = self.cam(imgs_to_rgb)
            # print(f"Shape of saliency maps: {saliency_maps.shape}")
            # print(f"Values in saliency maps: {torch.unique(saliency_maps)}")
            # print(
            #     f"Max and min in saliency maps: {torch.max(saliency_maps), torch.min(saliency_maps)}"
            # )

            ### Using guided backprop for interpretability ###
            ### This is needed because guided_backprop requires batch_size to be equal to 1
            # saliency_maps = self.guided_backprop(imgs_to_rgb)
            # print(f"Shape of saliency maps: {saliency_maps.shape}")
            # print(f"Values in saliency maps: {torch.unique(saliency_maps)}")
            # print(
            #     f"Max and min in saliency maps: {torch.max(saliency_maps), torch.min(saliency_maps)}"
            # )

            # bin_saliency_maps = torch.where(saliency_maps > 0.5, 1.0, 0.0)
            # bin_saliency_maps_3_ch = K.color.grayscale_to_rgb(bin_saliency_maps)
            # # print(f"Shape of binary saliency maps: {saliency_maps.shape}")
            # opposite_saliency_maps = torch.where(
            #     bin_saliency_maps == 0.0, 1.0, 0.0
            # )  # no need to get opposite of cam
            # imgs_to_reconstruct = imgs * opposite_saliency_maps
            # imgs_to_reconstruct = imgs * bin_saliency_maps_3_ch

            # print(
            #     f"\nPrimo e secondo canale bin_saliency_maps: {bin_saliency_maps[:, 0, :, :].all() == bin_saliency_maps[:, 1, :, :].all()}"
            # )
            # print(
            #     f"\nPrimo e terzo canale bin_saliency_maps: {bin_saliency_maps[:, 0, :, :].all() == bin_saliency_maps[:, 2, :, :].all()}"
            # )
            # print(
            #     f"\nSecondo e terzo canale bin_saliency_maps: {bin_saliency_maps[:, 1, :, :].all() == bin_saliency_maps[:, 2, :, :].all()}"
            # )

            segmentation_preds = self.model(
                imgs, script_type="validation"
            )  # features shape = [8, 512, 32, 32]
            loss_seg = self.loss_segmentation(segmentation_preds, labels)

            loss = loss_seg
            # compute the dice score
            dice_score = torchmetrics.functional.dice(
                segmentation_preds,
                labels,
                zero_division=1,
                num_classes=2,
                ignore_index=0,
            )

            self.log("val_dice", dice_score, on_step=False, on_epoch=True)
            self.log("total_val_loss", loss, on_step=False, on_epoch=True)
            self.log("val_segmentation_loss", loss_seg, on_step=False, on_epoch=True)
            # self.log("val_recontruction_loss", loss_rec, on_step=False, on_epoch=True)
            self.log("RAM", psutil.virtual_memory().percent, on_step=True)
            return (
                loss,
                segmentation_preds,
                # opposite_saliency_maps,
            )  # Return tensor to call ".backward" on

    def on_validation_batch_end(
        self,
        outputs: Optional[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        imgs, labels, centre = batch
        (
            _,
            segmentation_preds,
            # opposite_saliency_maps,
        ) = outputs
        dice = torchmetrics.functional.dice(
            segmentation_preds, labels, zero_division=1, num_classes=2, ignore_index=0
        )
        segmentation_preds_to_plot = torch.argmax(segmentation_preds, 1)

        if self.current_epoch == (self.trainer.max_epochs - 1):
            if batch_idx % 10 == 0:
                images = [
                    [
                        centre.to(torch.float),
                        dice,
                        img,
                        # wandb.Image(opposite_saliency_map),
                        label.float(),
                        segmentation_pred.float(),
                    ]
                    for img, label, segmentation_pred in zip(
                        imgs,
                        # opposite_saliency_maps,
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
                row[1],
                wandb.Image(row[2]),
                wandb.Image(row[3]),
                wandb.Image(row[4]),
                wandb.Image(row[5]),
                wandb.Image(row[6]),
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
        return segmentation_preds

    def on_test_batch_end(
        self,
        outputs: Optional[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        imgs, labels, centre = batch
        segmentation_preds = outputs
        dice = torchmetrics.functional.dice(
            segmentation_preds, labels, zero_division=1, num_classes=2, ignore_index=0
        )
        preds_plot = torch.argmax(segmentation_preds, 1)
        images = [
            [
                centre.to(torch.float),
                dice,
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
