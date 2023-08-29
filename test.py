from NewSDNet.dataset import PolypsDataModule
from NewSDNet.utils.centres_names import *
from NewSDNet.utils.csv_dicts import csv_dict
from NewSDNet.models import (
    SDNet,
    SDNetLightning,
    SDNetGradCamLightning,
    ResNetLightning,
    UNet,
    GradCamUNetLightning,
    deeplabv3plus_resnet101,
    DeepLabLightning,
    GradCamDeepLabLightning,
)
import lightning.pytorch as pl
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import argparse
import random
import torch
import torchvision.models as models
import warnings

WANDB_CACHE_DIR = "/projects/SiH_disentanglement/"

warnings.filterwarnings("ignore")


def main(args):
    ### Seed everything ###
    random.seed(args.seed)
    pl.seed_everything(args.seed)

    ### Initialize logging ###
    wandb_logger = WandbLogger(name=args.wandb_name, project=args.project_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_path,
        save_top_k=1,
        save_last=True,
        monitor="val_dice",
        mode="max",
    )
    checkpoint_classifier = ModelCheckpoint(
        dirpath=args.save_path,
        save_top_k=1,
        save_last=True,
        monitor="Validation accuracy",
        mode="max",
    )
    ### Initialize LightningDataModule ###

    if (
        args.model_name == "sdnet_gradcam"
        or args.model_name == "gradcam_unet"
        or args.model_name == "gradcam_deeplab"
    ):
        polypsDataset = PolypsDataModule(
            imgs_centres=imgs_centres,
            seg_centres=seg_centres,
            imgs_out_test_centre=[imgs_test_centre],  # imgs_test_centre,
            seg_out_test_centre=[seg_test_centre],  # seg_test_centre,
            csv_file_name=args.csv_file_name,
            save_path=args.save_path,
            train_batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            from_csv=True,
            per_patient=False,
            path_to_csvs=csv_dict,
            percentage_train=args.percentage_train,
            load_saliency=True,
            csv_saliency=args.csv_saliency,
        )

    else:
        polypsDataset = PolypsDataModule(
            imgs_centres=imgs_centres,
            seg_centres=seg_centres,
            imgs_out_test_centre=[imgs_test_centre],  # imgs_test_centre,
            seg_out_test_centre=[seg_test_centre],  # seg_test_centre,
            csv_file_name=args.csv_file_name,
            save_path=args.save_path,
            train_batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            from_csv=True,
            per_patient=False,
            path_to_csvs=csv_dict,
            percentage_train=args.percentage_train,
        )
    ### Initialize model ###
    sdnet_params = {
        "width": 512,
        "height": 512,
        "ndf": 64,
        "norm": "batchnorm",
        "upsample": "nearest",
        "num_classes": 1,
        "anatomy_out_channels": args.anatomy_factors,
        "z_lenght": args.modality_factors,
        "num_mask_channels": 8,
    }

    if args.model_name == "classifier":
        ResNetLight = ResNetLightning.load_from_checkpoint(
            checkpoint_path=args.ckpt_path,
            num_classes=5,
            lr=args.learning_rate,
            img_logger=wandb_logger,
            batch_size=args.train_batch_size,
            resnet_version=50,
            transfer=True,
        )

        torch.save(
            ResNetLight.model.state_dict(),
            "/projects/SiH_disentanglement/results/classifier_no_sobel_no_centre_5/epoch=3-step=1072_state_dict.ckpt",
        )

    if args.model_name == "sdnet":
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
        SDNetLight = SDNetLightning.load_from_checkpoint(
            checkpoint_path=args.ckpt_path,
            model=model,
            lr=args.learning_rate,
            img_logger=wandb_logger,
            batch_size=args.train_batch_size,
            weight_init=args.weight_init,
            charbonnier=args.charbonnier,
            reco_w=args.reco_w,
            kl_w=args.kl_w,
            dice_w=args.dice_w,
            regress_w=args.regress_w,
            focal_w=args.focal_w,
        )

    if args.model_name == "sdnet_gradcam":
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
        SDNetGradCamLight = SDNetGradCamLightning.load_from_checkpoint(
            checkpoint_path=args.ckpt_path,
            model=model,
            classifier=models.resnet50(),
            classifier_ckpt_path=args.classifier_ckpt_path,
            lr=args.learning_rate,
            img_logger=wandb_logger,
            batch_size=args.train_batch_size,
            weight_init=args.weight_init,
            charbonnier=args.charbonnier,
            reco_w=args.reco_w,
            kl_w=args.kl_w,
            dice_w=args.dice_w,
            regress_w=args.regress_w,
            focal_w=args.focal_w,
        )

    if args.model_name == "gradcam_unet":
        model = UNet(n_channels=3, n_classes=2, bilinear=True)
        GradCamUNetLight = GradCamUNetLightning.load_from_checkpoint(
            checkpoint_path=args.ckpt_path,
            model=model,
            lr=args.learning_rate,
            img_logger=wandb_logger,
            batch_size=args.train_batch_size,
            save_path=args.save_path,
        )
    if args.model_name == "deeplab":
        model_map = {"deeplabv3plus_resnet101": deeplabv3plus_resnet101}
        model = model_map["deeplabv3plus_resnet101"](
            num_classes=2, output_stride=16, pretrained_backbone=False
        )
        DeepLabLight = DeepLabLightning.load_from_checkpoint(
            checkpoint_path=args.ckpt_path,
            model=model,
            lr=args.learning_rate,
            img_logger=wandb_logger,
            batch_size=args.train_batch_size,
            save_path=args.save_path,
        )
    if args.model_name == "gradcam_deeplab":
        model_map = {"deeplabv3plus_resnet101": deeplabv3plus_resnet101}
        model = model_map["deeplabv3plus_resnet101"](
            num_classes=2, output_stride=16, pretrained_backbone=False
        )
        GradCamDeepLabLight = GradCamDeepLabLightning.load_from_checkpoint(
            checkpoint_path=args.ckpt_path,
            model=model,
            lr=args.learning_rate,
            img_logger=wandb_logger,
            batch_size=args.train_batch_size,
            save_path=args.save_path,
        )
    ### Testing ###

    if args.model_name == "sdnet":
        trainer = Trainer(
            max_epochs=args.max_epochs,
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )
        trainer.test(model=SDNetLight, datamodule=polypsDataset)
    # elif args.model_name == "classifier":
    #     trainer = Trainer(
    #         max_epochs=args.max_epochs,
    #         num_sanity_val_steps=1,
    #         logger=wandb_logger,
    #         callbacks=[checkpoint_classifier],
    #         inference_mode=False,
    #     )
    #     trainer.test(ResNetLight, datamodule=polypsDataset)

    elif args.model_name == "sdnet_gradcam":
        trainer = Trainer(
            max_epochs=args.max_epochs,
            num_sanity_val_steps=0,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )

        trainer.test(SDNetGradCamLight, datamodule=polypsDataset)

    elif args.model_name == "gradcam_unet":
        trainer = Trainer(
            max_epochs=args.max_epochs,
            num_sanity_val_steps=0,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )

        trainer.test(GradCamUNetLight, datamodule=polypsDataset)
    elif args.model_name == "deeplab":
        trainer = Trainer(
            max_epochs=args.max_epochs,
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )
        trainer.test(DeepLabLight, datamodule=polypsDataset)
    elif args.model_name == "gradcam_deeplab":
        trainer = Trainer(
            max_epochs=args.max_epochs,
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )
        trainer.test(GradCamDeepLabLight, datamodule=polypsDataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get all command line arguments")

    ### Training parameters ###
    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=1e-5,  # 1e-5
        help="Specify the initial learning rate",
    )
    parser.add_argument(
        "--max-epochs",
        dest="max_epochs",
        type=int,
        default=300,
        help="Specify the number of max epochs for training",
    )
    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        default=2,
        help="Specify number of workers",
    )

    # parser.add_argument(
    #     "--from-csv",
    #     dest="from_csv",
    #     default=False,
    #     action="store_true",
    #     help="Specify if you want to load dataset splits from csv files",
    # )

    parser.add_argument(
        "--train-batch-size",
        dest="train_batch_size",
        type=int,
        default=4,
        help="Specify batch size for training step",
    )
    parser.add_argument(
        "--percentage-train",
        dest="percentage_train",
        type=int,
        default=0.8,
        help="Specify percentage of training data",
    )
    parser.add_argument(
        "--ckpt-path",
        dest="ckpt_path",
        type=str,
        default=None,
        help="Path to checkpoint file",
    )

    parser.add_argument(
        "--csv-saliency",
        dest="csv_saliency",
        type=str,
        help="Specify path to centre saliency maps if needed",
    )
    ### Initialization parameters ###
    parser.add_argument(
        "--model-name",
        dest="model_name",
        type=str,
        default="baseline",
        help="Specify the model to be used",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=1,
        help="Specify the global random seed",
    )
    parser.add_argument(
        "--save-path",
        dest="save_path",
        type=str,
        help="Specify path to folder where results are saved",
    )

    parser.add_argument(
        "--classifier-ckpt-path",
        dest="classifier_ckpt_path",
        type=str,
        help="Specify path to centre classifier checkpoint",
    )

    parser.add_argument(
        "--csv-file-name",
        dest="csv_file_name",
        type=str,
        help="Specify appendix for csv splits name (i.e. give name of experiment)",
    )

    parser.add_argument(
        "--wandb-name",
        dest="wandb_name",
        type=str,
        help="Specify the name of the W&B run",
    )

    parser.add_argument(
        "--project-name",
        dest="project_name",
        type=str,
        help="Specify the name of the W&B project",
    )

    ### SDNet parameters ###
    parser.add_argument(
        "--anatomy-factors",
        dest="anatomy_factors",
        type=int,
        default=8,
        help="Specify number of anatomy factors to encode",
    )
    parser.add_argument(
        "--modality-factors",
        dest="modality_factors",
        type=int,
        default=8,
        help="Specify number of modality factors to encode",
    )
    parser.add_argument(
        "--charbonnier",
        type=int,
        default=0,
        help="Specify Charbonnier penalty for the reconstruction loss",
    )
    parser.add_argument(
        "--kl-w",
        dest="kl_w",
        type=float,
        default=0.01,
        help="KL divergence loss weight",
    )
    parser.add_argument("--dice_w", type=float, default=10.0, help="Dice loss weight")
    parser.add_argument(
        "--reco-w",
        dest="reco_w",
        type=float,
        default=1.0,
        help="Reconstruction loss weight",
    )
    parser.add_argument(
        "--weight-init",
        dest="weight_init",
        type=str,
        default="xavier",
        help="Weight initialization method",
    )
    parser.add_argument(
        "--regress-w",
        dest="regress_w",
        type=float,
        default=1.0,
        help="Regression loss weight",
    )
    parser.add_argument(
        "--focal-w", dest="focal_w", type=float, default=0.0, help="Focal loss weight"
    )

    args = parser.parse_args()

    main(args)
