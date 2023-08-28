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


class PolypClassifierPreTrainLightning(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        img_logger: pl.Callback,
        batch_size: int,
    ):
        super().__init__()
        # Deactivate automatic optimization for adversarial training

        # self.save_hyperparameters()
        # self.automatic_optimization = False

        self.lr = lr
        self.batch_size = batch_size
        self.img_logger = img_logger
        self.cross_entropy = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="binary", num_classes=2)
        self.train_table = []
        self.columns_train_table = [
            "centre",
            "polyp presence predictions",
            "ground_truth",
            "image",
        ]
        self.val_table = []
        self.columns_val_table = [
            "accuracy",
            "centre",
            "polyp presence predictions",
            "ground_truth",
            "image",
        ]
        self.anatomy_in_test = []
        self.anatomy_out_test = []
        self.in_test_table = []

        self.polyp_detector = resnet50()
        self.polyp_detector.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        linear_size = list(self.polyp_detector.children())[-1].in_features
        self.polyp_detector.fc = nn.Linear(
            linear_size, 2
        )  # 2 like the number of classes: polyp/no-polyp

    def forward(self, labels):
        return self.polyp_detector(labels)

    def configure_optimizers(self) -> Any:
        # opt_sdnet = optim.AdamW(self.sdnet.parameters(), lr=self.lr)
        # opt_classifier = optim.AdamW(self.classifier.parameters(), lr=self.lr)
        # self.parameters

        return optim.AdamW(self.polyp_detector.parameters(), lr=self.lr)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        imgs, labels, centres = batch
        # Compute labels for polyp_calssifier
        polyp_labels = []
        for label in labels:
            label_values = torch.unique(label)
            if 1 in label_values:
                polyp_labels.append(1)
            else:
                polyp_labels.append(0)
        # labels_float = torch.tensor(labels, dtype=torch.float)
        polyp_labels = torch.tensor(polyp_labels, dtype=torch.float).to(device="cuda")
        polyp_presence_predictions = self.polyp_detector(
            labels.unsqueeze(1).to(torch.float)
        )
        # polyp_presence_predictions = (
        #     torch.argmax(polyp_presence_predictions, 1)
        # )
        polyp_presence_loss = self.cross_entropy(
            polyp_presence_predictions,
            polyp_labels,
        )
        # Compute dice score for segmentations
        accuracy = self.accuracy(polyp_presence_predictions, polyp_labels)
        # logging
        self.log(
            "Polyp presence detector loss",
            polyp_presence_loss,
            on_step=False,
            on_epoch=True,
        )

        # Plot reconstruction, anatomy channels and segmentations in W&B table
        if self.current_epoch == (self.trainer.max_epochs - 1):
            if batch_idx == 5:
                general_imgs = [
                    [
                        centre.float(),
                        polyp_presence_pred.float(),
                        gt.float(),
                        img,
                    ]
                    for centre, polyp_presence_pred, gt, img in zip(
                        centres,
                        polyp_presence_predictions,
                        labels,
                        imgs,
                    )
                ]
                self.train_table.extend(general_imgs)

        self.log("accuracy training", accuracy, on_step=False, on_epoch=True)
        return polyp_presence_loss

    def validation_step(self, batch, batch_idx) -> Optional[Any]:
        imgs, labels, centres = batch
        # Compute labels for polyp_calssifier
        polyp_labels = []
        for label in labels:
            label_values = torch.unique(label)
            if 1 in label_values:
                polyp_labels.append(1)
            else:
                polyp_labels.append(0)
        # labels_float = torch.tensor(labels, dtype=torch.float)
        polyp_labels = (
            torch.tensor(polyp_labels, dtype=int)
            .type(torch.LongTensor)
            .to(device="cuda")
        )
        polyp_presence_predictions = self.polyp_detector(
            labels.unsqueeze(1).to(torch.float)
        )
        # polyp_presence_predictions = torch.argmax(polyp_presence_predictions, 1).to(
        #     torch.float
        # )
        print(
            f"\nPOLYP PRESENCE PREDICTIONS SHAPE, TYPE E DTYPE: {polyp_presence_predictions, polyp_presence_predictions.shape, polyp_presence_predictions.dtype, polyp_presence_predictions.type}"
        )
        print(
            f"\nPOLYP LABELS SHAPE, TYPE E DTYPE: {polyp_labels, polyp_labels.shape, polyp_labels.dtype, polyp_labels.type}"
        )
        polyp_presence_loss = self.cross_entropy(
            polyp_presence_predictions, polyp_labels
        )
        # Compute dice score for segmentations
        accuracy = self.accuracy(polyp_presence_predictions, polyp_labels)
        # Plot reconstruction, anatomy channels and segmentations in W&B table
        if self.current_epoch == (self.trainer.max_epochs - 1):
            if batch_idx == 5:
                general_imgs = [
                    [
                        centre.float(),
                        accuracy.float(),
                        polyp_presence_pred.float(),
                        gt.float(),
                        img,
                    ]
                    for centre, polyp_presence_pred, gt, img in zip(
                        centres,
                        polyp_presence_predictions,
                        labels,
                        imgs,
                    )
                ]
                self.val_table.extend(general_imgs)

        self.log("Validation accuracy", accuracy, on_step=False, on_epoch=True)
        self.log("loss validation", polyp_presence_loss, on_step=False, on_epoch=True)

    def on_train_end(self) -> None:
        train_table_wandb = []
        for row in self.train_table:
            anatomy_wand = [
                row[0],
                row[1],
                wandb.Image(row[2]),
                wandb.Image(row[3]),
            ]

        val_table_wandb = []
        for row in self.val_table:
            anatomy_wand = [
                row[0],
                row[1],
                row[2],
                wandb.Image(row[3]),
                wandb.Image(row[4]),
            ]
            val_table_wandb.append(anatomy_wand)

        self.trainer.logger.log_table(
            key="train general table",
            columns=self.columns_train_table,
            data=train_table_wandb,
        )

        self.trainer.logger.log_table(
            key="validation general table",
            columns=self.columns_val_table,
            data=val_table_wandb,
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> Optional[Any]:
        imgs, labels, centres = batch
        # Compute labels for polyp_calssifier
        polyp_labels = []
        for label in labels:
            label_values = torch.unique(label)
            if 1 in label_values:
                polyp_labels.append(1)
            else:
                polyp_labels.append(0)
        labels_float = torch.tensor(labels, dtype=torch.float)
        polyp_labels = torch.tensor(polyp_labels, dtype=torch.float)
        polyp_presence_predictions = self.polyp_detector(labels_float.unsqueeze(1))
        polyp_presence_predictions = torch.argmax(polyp_presence_predictions, 1).to(
            torch.float
        )
        # Compute dice score for segmentations
        accuracy = self.accuracy(polyp_presence_predictions, polyp_labels)

        self.log(
            f"{dataloader_idx}_test_accuracy", accuracy, on_step=False, on_epoch=True
        )

    def on_test_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        imgs, labels, centres = batch
        # Compute labels for polyp_calssifier
        polyp_labels = []
        for label in labels:
            label_values = torch.unique(label)
            if 1 in label_values:
                polyp_labels.append(1)
            else:
                polyp_labels.append(0)
        labels_float = torch.tensor(labels, dtype=torch.float)
        polyp_labels = torch.tensor(polyp_labels, dtype=torch.int)
        polyp_presence_predictions = self.polyp_detector(labels_float.unsqueeze(1))
        # Compute dice score for segmentations
        accuracy = self.accuracy(polyp_presence_predictions, polyp_labels)
        polyp_presence_predictions = torch.argmax(polyp_presence_predictions, 1)

        general_imgs = [
            [
                accuracy.float(),
                centre.float(),
                polyp_presence_pred.float(),
                gt.float(),
                img,
            ]
            for centre, polyp_presence_pred, gt, img in zip(
                centres, polyp_presence_predictions, labels, imgs
            )
        ]

        self.in_test_table.extend(general_imgs)

    def on_test_end(self) -> None:
        in_test_wandb = []
        for row in self.in_test_table:
            anatomy_wand = [
                row[0],
                row[1],
                row[2],
                wandb.Image(row[3]),
                wandb.Image(row[4]),
            ]
            in_test_wandb.append(anatomy_wand)

        self.trainer.logger.log_table(
            key="in distribution test table",
            columns=self.columns_test_table,
            data=in_test_wandb,
        )
