import json
from NewSDNet.dataset import PolypsDataModule
from NewSDNet.utils.centres_names import *
from NewSDNet.utils.csv_dicts import csv_dict
from NewSDNet.models import (
    SDNet,
    SDNetLightning,
    SDNetGradCamLightning,
    ResNetLightning,
    UNet,
    UNetLightning,
    GradCamUNetLightning,
    deeplabv3plus_resnet101,
    DeepLabLightning,
    GradCamDeepLabLightning,
)
import lightning.pytorch as pl
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import random
import torchvision.models as models
import warnings
import hydra
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings("ignore")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    print(OmegaConf.to_yaml(args))
    ### Seed everything ###
    random.seed(args["seed"])
    pl.seed_everything(args["seed"])

    ### Initialize logging ###
    wandb_logger = WandbLogger(
        name=args["wandb_name"],
        project=args["project_name"],
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=args["save_path"],
        save_top_k=1,
        save_last=True,
        monitor="val_dice",
        mode="max",
    )
    checkpoint_classifier = ModelCheckpoint(
        dirpath=args["save_path"],
        save_top_k=1,
        save_last=True,
        monitor="Validation accuracy",
        mode="max",
    )

    # resolve out distribution test centre
    imgs_test_centre = [
        {center_string: imgs_centres.pop(center_string)}
        for center_string in args["out_centres"]
    ]
    seg_test_centre = [
        {center_string: seg_centres.pop(center_string)}
        for center_string in args["out_centres"]
    ]
    # !!! TEST THAT POP ACTUALLY REMOVES

    print(f"imgs_test_centre: {imgs_test_centre}\n")
    print(f"seg_test_centre: {seg_test_centre}\n")
    print(f"imgs_centres: {imgs_centres}\n")
    print(f"seg_centres: {seg_centres}\n")
    ### Initialize LightningDataModule ###
    if (
        args["model_name"] == "sdnet_gradcam"
        or args["model_name"] == "gradcam_unet"
        or args["model_name"] == "gradcam_deeplab"
    ):
        polypsDataset = PolypsDataModule(
            imgs_centres=imgs_centres,
            seg_centres=seg_centres,
            imgs_out_test_centre=imgs_test_centre,
            seg_out_test_centre=seg_test_centre,
            csv_file_name=args["csv_file_name"],
            save_path=args["save_path"],
            train_batch_size=args["train_batch_size"],
            num_workers=args["num_workers"],
            seed=args["seed"],
            from_csv=True,
            per_patient=False,
            path_to_csvs=csv_dict,
            percentage_train=args["percentage_train"],
            load_saliency=True,
            csv_saliency=args["csv_saliency"],
            flag_no_centres=args["out_centres"] == [],
        )

    else:
        polypsDataset = PolypsDataModule(
            imgs_centres=imgs_centres,
            seg_centres=seg_centres,
            imgs_out_test_centre=imgs_test_centre,
            seg_out_test_centre=seg_test_centre,
            csv_file_name=args["csv_file_name"],
            save_path=args["save_path"],
            train_batch_size=args["train_batch_size"],
            num_workers=args["num_workers"],
            seed=args["seed"],
            from_csv=True,
            per_patient=False,
            path_to_csvs=csv_dict,
            percentage_train=args["percentage_train"],
            flag_no_centres=args["out_centres"] == [],
        )

    ### Initialize model ###
    sdnet_params = {
        "width": 512,
        "height": 512,
        "ndf": 64,
        "norm": "batchnorm",
        "upsample": "nearest",
        "num_classes": 1,
        "anatomy_out_channels": args["anatomy_factors"],
        "z_lenght": args["modality_factors"],
        "num_mask_channels": 8,
    }

    if args["model_name"] == "sdnet_gradcam":
        model = SDNet(
            sdnet_params["width"],
            sdnet_params["height"],
            sdnet_params["num_classes"],
            sdnet_params["ndf"],
            sdnet_params["z_lenght"],
            sdnet_params["norm"],
            sdnet_params["upsample"],
            sdnet_params["anatomy_out_channels"],
            sdnet_params["num_mask_channels"],
        )
        SDNetGradCamLight = SDNetGradCamLightning(
            model=model,
            classifier=models.resnet50(),
            classifier_ckpt_path=args["classifier_ckpt_path"],
            lr=args["learning_rate"],
            img_logger=wandb_logger,
            batch_size=args["train_batch_size"],
            weight_init=args["weight_init"],
            charbonnier=args["charbonnier"],
            reco_w=args["reco_w"],
            kl_w=args["kl_w"],
            dice_w=args["dice_w"],
            regress_w=args["regress_w"],
            focal_w=args["focal_w"],
        )

    if args["model_name"] == "classifier":
        ResNetLight = ResNetLightning(
            num_classes=5,
            lr=args["learning_rate"],
            img_logger=wandb_logger,
            batch_size=args["train_batch_size"],
            resnet_version=50,
            transfer=True,
            fine_tuning=False,
        )

    if args["model_name"] == "sdnet":
        model = SDNet(
            sdnet_params["width"],
            sdnet_params["height"],
            sdnet_params["num_classes"],
            sdnet_params["ndf"],
            sdnet_params["z_lenght"],
            sdnet_params["norm"],
            sdnet_params["upsample"],
            sdnet_params["anatomy_out_channels"],
            sdnet_params["num_mask_channels"],
        )
        SDNetLight = SDNetLightning(
            model=model,
            lr=args["learning_rate"],
            img_logger=wandb_logger,
            batch_size=args["train_batch_size"],
            weight_init=args["weight_init"],
            charbonnier=args["charbonnier"],
            reco_w=args["reco_w"],
            kl_w=args["kl_w"],
            dice_w=args["dice_w"],
            regress_w=args["regress_w"],
            focal_w=args["focal_w"],
        )

    if args["model_name"] == "unet":
        model = UNet(n_channels=3, n_classes=2, bilinear=True)
        UNetLight = UNetLightning(
            model=model,
            lr=args["learning_rate"],
            img_logger=wandb_logger,
            save_path=args["save_path"],
        )

    if args["model_name"] == "gradcam_unet":
        model = UNet(n_channels=3, n_classes=2, bilinear=True)
        GradCamUNetLight = GradCamUNetLightning(
            model=model,
            lr=args["learning_rate"],
            img_logger=wandb_logger,
            batch_size=args["train_batch_size"],
            save_path=args["save_path"],
        )
    if args["model_name"] == "gradcam_deeplab":
        model_map = {"deeplabv3plus_resnet101": deeplabv3plus_resnet101}
        model = model_map["deeplabv3plus_resnet101"](
            num_classes=2, output_stride=16, pretrained_backbone=False
        )
        GradCamDeepLabLight = GradCamDeepLabLightning(
            model=model,
            lr=args["learning_rate"],
            img_logger=wandb_logger,
            batch_size=args["train_batch_size"],
            save_path=args["save_path"],
        )
    if args["model_name"] == "deeplab":
        model_map = {"deeplabv3plus_resnet101": deeplabv3plus_resnet101}
        model = model_map["deeplabv3plus_resnet101"](
            num_classes=2, output_stride=16, pretrained_backbone=False
        )
        DeepLabLight = DeepLabLightning(
            model=model,
            lr=args["learning_rate"],
            img_logger=wandb_logger,
            batch_size=args["train_batch_size"],
            save_path=args["save_path"],
        )

    ### Training ###

    if args["model_name"] == "sdnet":
        trainer = Trainer(
            max_epochs=args["max_epochs"],
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )

        trainer.fit(
            SDNetLight,
            datamodule=polypsDataset,
            # ckpt_path="/projects/shift_review/sdnet_miccai/projects/shift_review/sdnet_miccai/sdnet_no_centre_4_best/last.ckpt",
        )
        trainer.test(SDNetLight, datamodule=polypsDataset, ckpt_path="best")

    elif args["model_name"] == "classifier":
        trainer = Trainer(
            max_epochs=args["max_epochs"],
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_classifier],
            inference_mode=False,
        )

        trainer.fit(
            ResNetLight,
            datamodule=polypsDataset,
            # ckpt_path="/projects/shift_review/sdnet_miccai/results/classifier_no_centre_5/last-v1.ckpt",
        )
        trainer.test(ResNetLight, datamodule=polypsDataset)

    elif args["model_name"] == "sdnet_gradcam":
        trainer = Trainer(
            max_epochs=args["max_epochs"],
            num_sanity_val_steps=0,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(
            SDNetGradCamLight,
            datamodule=polypsDataset,
            # ckpt_path="/projects/SiH_disentanglement/results/sdnet_gradcam_aug_no_centre_3/last.ckpt",
        )
        trainer.test(SDNetGradCamLight, datamodule=polypsDataset, ckpt_path="best")

    elif args["model_name"] == "unet":
        trainer.fit(
            UNetLight,
            datamodule=polypsDataset,
            # ckpt_path="/processing/v.corbetta/ranking_loss/results/unet_baseline_no_centre4_best/last-v1.ckpt",
        )
        trainer.test(UNetLight, datamodule=polypsDataset, ckpt_path="best")

    elif args["model_name"] == "gradcam_unet":
        trainer = Trainer(
            max_epochs=args["max_epochs"],
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(GradCamUNetLight, datamodule=polypsDataset)
        trainer.test(GradCamUNetLight, datamodule=polypsDataset, ckpt_path="best")

    elif args["model_name"] == "gradcam_deeplab":
        trainer = Trainer(
            max_epochs=args["max_epochs"],
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(GradCamDeepLabLight, datamodule=polypsDataset)
        trainer.test(GradCamDeepLabLight, datamodule=polypsDataset, ckpt_path="best")

    elif args["model_name"] == "deeplab":
        trainer = Trainer(
            max_epochs=args["max_epochs"],
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(DeepLabLight, datamodule=polypsDataset)
        trainer.test(DeepLabLight, datamodule=polypsDataset, ckpt_path="best")


if __name__ == "__main__":
    main()
