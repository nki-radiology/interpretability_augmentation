from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import wandb
import kornia as K
import monai
import torchvision.models as models
from monai.visualize import CAM
from monai.visualize.class_activation_maps import GradCAM
from captum.attr import LRP, visualization, DeepLift, LayerLRP, LayerDeepLift
import numpy as np
from pathlib import Path
import pandas as pd

# TODO: fix classifier with hydra so that we have two config files, one for training
# and one to generate the gradcam visualizations


class ResNetLightning(pl.LightningModule):
    resnets = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }

    def __init__(
        self,
        num_classes: int,
        lr: float,
        img_logger: pl.Callback,
        batch_size: int,
        resnet_version: int,
        transfer: bool = True,
        tune_fc_only: bool = False,
        fine_tuning: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.img_logger = img_logger
        self.fine_tuning = fine_tuning
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        self.test_table: list = []
        self.columns: list[str] = [
            "class",
            "prediction",
            "image",
            "sobel",
            "saliency map alone",
            "saliency map binarised",
            "saliency map",
            "cam map alone",
            "cam map binarised",
        ]
        self.model = self.resnets[resnet_version](pretrained=transfer)
        self.list_paths_saliency_maps = []
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.model.children())[-1].in_features
        self.model.fc = nn.Linear(linear_size, num_classes)

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self) -> Any:
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        imgs, _, centres = batch
        imgs_gray = K.color.rgb_to_grayscale(imgs)
        imgs_sobel: torch.Tensor = K.filters.sobel(imgs_gray)
        imgs_to_rgb: torch.Tensor = K.color.grayscale_to_rgb(imgs_sobel)

        outputs = self.model(imgs_to_rgb)
        loss = self.criterion(outputs, centres)

        # Compute accuracy score for predictions
        acc = self.accuracy(outputs, centres)

        self.log("Training loss", loss, on_step=False, on_epoch=True)
        self.log("Training accuracy", acc, on_step=False, on_epoch=True)

        return loss  # Return tensor to call ".backward()" on

    def validation_step(self, batch, batch_idx) -> Optional[Any]:
        imgs, _, centres = batch
        imgs_gray = K.color.rgb_to_grayscale(imgs)
        imgs_sobel: torch.Tensor = K.filters.sobel(imgs_gray)
        imgs_to_rgb: torch.Tensor = K.color.grayscale_to_rgb(imgs_sobel)

        outputs = self.model(imgs_to_rgb)
        loss = self.criterion(outputs, centres)

        # Compute accuracy score for predictions
        acc = self.accuracy(outputs, centres)

        self.log("Validation loss", loss, on_step=False, on_epoch=True)
        self.log("Validation accuracy", acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0) -> Optional[Any]:
        imgs, _, centres, _ = batch  # ,_
        imgs_gray = K.color.rgb_to_grayscale(imgs)
        imgs_sobel: torch.Tensor = K.filters.sobel(imgs_gray)
        imgs_to_rgb: torch.Tensor = K.color.grayscale_to_rgb(imgs_sobel)

        outputs = self.model(imgs_to_rgb)

        # Compute accuracy score for predictions
        acc = self.accuracy(outputs, centres)

        self.log(f"{dataloader_idx}_test_accuracy", acc, on_step=False, on_epoch=True)

    def on_test_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        imgs, _, centres = batch  # , imgs_path
        imgs_gray = K.color.rgb_to_grayscale(imgs)
        imgs_sobel: torch.Tensor = K.filters.sobel(imgs_gray)
        imgs_to_rgb: torch.Tensor = K.color.grayscale_to_rgb(imgs_sobel)

        with torch.enable_grad():
            outputs = self.model(imgs)  # imgs_to_rgb
            # children_layer4 = [child for child in list(self.model.layer4.children())]
            cam = CAM(
                nn_module=self.model, target_layers="layer4.2.relu", fc_layers="fc"
            )
            gradcam = GradCAM(nn_module=self.model, target_layers="layer4.2.conv3")
            gradcam_maps = gradcam(imgs)  # imgs_to_rgb
            # lrp = LRP(self.model)
            # classes_for_lrp = [torch.argmax(output).item() for output in outputs]
            # attribution = lrp.attribute(imgs_to_rgb, target=1)  # target=classes_for_lrp
            # layer_deeplift = LayerDeepLift(self.model, children_layer4[2].conv1)
            # print(list(self.model.layer4.children()))

            # print(f"\nAAAAAAAAAAAAAAA: {children_layer4[2].conv1}\n")
            # layer_lrp = LayerLRP(self.model, children_layer4[2].conv1)
            # attribution_layer_lrp = layer_lrp.attribute(
            #     imgs_to_rgb, target=classes_for_lrp
            # )
            # attribution_layer_deeplift = layer_deeplift.attribute(
            #     imgs_to_rgb, target=classes_for_lrp
            # )
            saliency_maps_calculator = (
                monai.visualize.gradient_based.GuidedBackpropGrad(self.model)
            )
            saliency_maps = saliency_maps_calculator(imgs)  # imgs_to_rgb
            saliency_maps_bin = torch.where(saliency_maps > 0.5, 1.0, 0.0)
            cam_maps = cam(imgs)  # imgs_to_rgb
            cam_maps_bin = torch.where(cam_maps > 0.5, 1.0, 0.0)
            # torch.save(
            #     cam_maps_bin,
            #     Path(
            #         "/processing/v.corbetta/saliency_train_no_centre_4/"
            #         + str(Path(imgs_path[0][0]).stem)
            #         + ".pt"
            #     ),
            # )

            # self.list_paths_saliency_maps.append(
            #     "/processing/v.corbetta/saliency_train_no_centre_4/"
            #     + str(Path(imgs_path[0][0]).stem)
            #     + ".pt"
            # )

            # fig_attr, _ = visualization.visualize_image_attr(
            #     np.transpose(
            #         attribution[0].squeeze().cpu().detach().numpy(), (1, 2, 0)
            #     ),
            #     np.transpose(imgs[0].squeeze().cpu().detach().numpy(), (1, 2, 0)),
            #     method="heat_map",
            #     show_colorbar=True,
            #     sign="all",
            # )

            # fig_attr_layer_lrp, _ = visualization.visualize_image_attr(
            #     np.transpose(
            #         attribution_layer_lrp[0].squeeze().cpu().detach().numpy(), (1, 2, 0)
            #     ),
            #     np.transpose(imgs[0].squeeze().cpu().detach().numpy(), (1, 2, 0)),
            #     method="heat_map",
            #     show_colorbar=True,
            #     sign="all",
            # )

            # fig_attr_layer_deeplift, _ = visualization.visualize_image_attr(
            #     np.transpose(
            #         attribution_layer_deeplift[0].squeeze().cpu().detach().numpy(),
            #         (1, 2, 0),
            #     ),
            #     np.transpose(imgs[0].squeeze().cpu().detach().numpy(), (1, 2, 0)),
            #     method="heat_map",
            #     show_colorbar=True,
            #     sign="all",
            # )

            imgs_to_log = [
                [
                    # centre,
                    # output,
                    wandb.Image(img),
                    wandb.Image(cam_map_bin.float()),
                    wandb.Image(cam_map.float()),
                ]
                for img, cam_map_bin, cam_map in zip(imgs, cam_maps_bin, cam_maps)
            ]

            # imgs_to_log = [
            #     [
            #         centre,
            #         torch.argmax(pred, 0),
            #         wandb.Image(img),
            #         wandb.Image(img_to_rgb.float()),
            #         wandb.Image(saliency_map.float()),
            #         wandb.Image(saliency_map_bin.float()),
            #         wandb.Image(
            #             monai.visualize.utils.blend_images(
            #                 img,
            #                 K.color.rgb_to_grayscale(saliency_map),
            #                 alpha=0.5,
            #                 cmap="RdYlGn",
            #                 rescale_arrays=True,
            #                 transparent_background=True,
            #             )
            #         ),
            #         wandb.Image(cam_map.float()),
            #         wandb.Image(cam_map_bin.float()),
            #         # wandb.Image(gradcam_map),
            #         # wandb.Image(fig_attr),
            #         # wandb.Image(fig_attr_layer_lrp),
            #         # wandb.Image(fig_attr_layer_deeplift),
            #     ]
            #     for centre, pred, img, img_to_rgb, saliency_map, saliency_map_bin, cam_map, cam_map_bin in zip(
            #         centres,
            #         outputs,
            #         imgs,
            #         imgs_to_rgb,
            #         saliency_maps,
            #         saliency_maps_bin,
            #         cam_maps,
            #         cam_maps_bin,
            #     )
            # ]

        self.test_table.extend(imgs_to_log)

    def on_test_end(self) -> None:
        self.trainer.logger.log_table(
            key="test results",
            # columns=self.columns,
            columns=[
                # "centre",
                # "pred_centre",
                "original image",
                "gradcam map bin",
                "gradcam map",
            ],
            data=self.test_table,
        )
        # pd.DataFrame(self.list_paths_saliency_maps, columns=["gradcam maps"]).to_csv(
        #     "/projects/shift_review/sdnet_miccai/results/saliency_train_no_centre_4.csv"
        # )
