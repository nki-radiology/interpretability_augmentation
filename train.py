from NewSDNet.dataset import PolypsDataModule
from NewSDNet.utils.centres_names import *
from NewSDNet.utils.csv_dicts import csv_dict
from NewSDNet.models import (
    SDNet,
    SDNetLightning,
    SDNetGradCamLightning,
    ResNetLightning,
    ModifiedUNet,
    ModifiedUNetLightning,
    SDNetStyleNonAdvLightning,
    SDNetStyleNonAdv,
    SDNetAnatomyDisentangler,
    SDNetAnatomyDisLight,
    GANSDNetLightning,
    SiHUNet,
    SiHUNetLightning,
    PolypClassifierLightning,
    UNet,
    GradCamUNetLightning,
    PolypClassifierPreTrainLightning,
    GANSDNetGradCamLightning,
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
import torchvision.models as models
import warnings

warnings.filterwarnings("ignore")


def main(args):
    ### Seed everything ###
    random.seed(args.seed)
    pl.seed_everything(args.seed)

    ### Initialize logging ###
    wandb_logger = WandbLogger(
        name=args.wandb_name,
        project=args.project_name,
    )
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
        or args.model_name == "modified_unet"
        or args.model_name == "gradcam_unet"
        or args.model_name == "gan_sdnet_gradcam"
        or args.model_name == "gradcam_deeplab"
    ):
        polypsDataset = PolypsDataModule(
            imgs_centres=imgs_centres,
            seg_centres=seg_centres,
            imgs_out_test_centre=[],  # imgs_test_centre,
            seg_out_test_centre=[],  # seg_test_centre,
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
            imgs_out_test_centre=[],  # imgs_test_centre,
            seg_out_test_centre=[],  # seg_test_centre,
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

    sdnet_classifier_params = {
        "width": 512,
        "height": 512,
        "ndf": 64,
        "norm": "batchnorm",
        "upsample": "nearest",
        "num_classes": 1,
        "num_centres": 5,
        "anatomy_out_channels": args.anatomy_factors,
        "z_lenght": args.modality_factors,
        "num_mask_channels": 8,
    }
    if args.model_name == "polyp_detector":
        PolypDetectorLight = PolypClassifierPreTrainLightning(
            lr=args.learning_rate,
            img_logger=wandb_logger,
            batch_size=args.train_batch_size,
        )
    if args.model_name == "polyp_classifier":
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
        PolypClassifierLight = PolypClassifierLightning(
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
    if args.model_name == "sihunet":
        model = SiHUNet(n_channels=3, n_classes=2, bilinear=True)
        SiHUNetLight = SiHUNetLightning(
            model=model,
            lr=args.learning_rate,
            img_logger=wandb_logger,
            batch_size=args.train_batch_size,
            save_path=args.save_path,
            q=0.7,
        )
    if args.model_name == "gan_sdnet":
        GANSDNetLight = GANSDNetLightning(
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
            z_lenght=sdnet_params["z_lenght"],
            height=sdnet_params["height"],
            width=sdnet_params["width"],
            ndf=sdnet_params["ndf"],
            anatomy_out_channels=sdnet_params["anatomy_out_channels"],
            upsample=sdnet_params["upsample"],
            norm=sdnet_params["norm"],
            num_mask_channels=sdnet_params["num_mask_channels"],
            classifier_ckpt=args.classifier_ckpt_path,
        )

    if args.model_name == "gan_sdnet_gradcam":
        GANSDNetGradCamLight = GANSDNetGradCamLightning(
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
            z_lenght=sdnet_params["z_lenght"],
            height=sdnet_params["height"],
            width=sdnet_params["width"],
            ndf=sdnet_params["ndf"],
            anatomy_out_channels=sdnet_params["anatomy_out_channels"],
            upsample=sdnet_params["upsample"],
            norm=sdnet_params["norm"],
            num_mask_channels=sdnet_params["num_mask_channels"],
            classifier_ckpt=args.classifier_ckpt_path,
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
        SDNetGradCamLight = SDNetGradCamLightning(
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

    if args.model_name == "modified_unet":
        model = ModifiedUNet(n_channels=3, n_classes=2, bilinear=True)
        modifiedUNetLight = ModifiedUNetLightning(
            model=model,
            # classifier=models.resnet50(),
            # classifier_ckpt_path=args.classifier_ckpt_path,
            lr=args.learning_rate,
            img_logger=wandb_logger,
            batch_size=args.train_batch_size,
            save_path=args.save_path,
        )
    if args.model_name == "classifier":
        ResNetLight = ResNetLightning(
            num_classes=5,
            lr=args.learning_rate,
            img_logger=wandb_logger,
            batch_size=args.train_batch_size,
            resnet_version=50,
            transfer=True,
            fine_tuning=False,
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
        SDNetLight = SDNetLightning(
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

    if args.model_name == "sdnet_style_nonadv":
        model = SDNetStyleNonAdv(
            sdnet_classifier_params["width"],
            sdnet_classifier_params["height"],
            sdnet_classifier_params["num_classes"],
            sdnet_classifier_params["num_centres"],
            sdnet_classifier_params["ndf"],
            sdnet_classifier_params["z_lenght"],
            sdnet_classifier_params["norm"],
            sdnet_classifier_params["upsample"],
            sdnet_classifier_params["anatomy_out_channels"],
            sdnet_classifier_params["num_mask_channels"],
        )
        SDNetStyleNonADVLight = SDNetStyleNonAdvLightning(
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
    if args.model_name == "sdnet_anatomy_dis":
        model = SDNetAnatomyDisentangler(
            sdnet_classifier_params["width"],
            sdnet_classifier_params["height"],
            sdnet_classifier_params["num_classes"],
            sdnet_classifier_params["num_centres"],
            sdnet_classifier_params["ndf"],
            sdnet_classifier_params["z_lenght"],
            sdnet_classifier_params["norm"],
            sdnet_classifier_params["upsample"],
            sdnet_classifier_params["anatomy_out_channels"],
            sdnet_classifier_params["num_mask_channels"],
        )
        SDNetAnatomyDis = SDNetAnatomyDisLight(
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

    if args.model_name == "gradcam_unet":
        model = UNet(n_channels=3, n_classes=2, bilinear=True)
        GradCamUNetLight = GradCamUNetLightning(
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
        GradCamDeepLabLight = GradCamDeepLabLightning(
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
        DeepLabLight = DeepLabLightning(
            model=model,
            lr=args.learning_rate,
            img_logger=wandb_logger,
            batch_size=args.train_batch_size,
            save_path=args.save_path,
        )

    ### Training ###

    if args.model_name == "sdnet":
        trainer = Trainer(
            max_epochs=args.max_epochs,
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

    elif args.model_name == "classifier":
        trainer = Trainer(
            max_epochs=args.max_epochs,
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

    elif args.model_name == "modified_unet":
        trainer = Trainer(
            max_epochs=args.max_epochs,
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
            inference_mode=False,
        )
        trainer.fit(
            modifiedUNetLight,
            datamodule=polypsDataset,
            # ckpt_path="/processing/v.corbetta/sdnet_miccai/results/modified_unet_no_centre_5_gradcam_best/last.ckpt",
        )
        trainer.test(modifiedUNetLight, datamodule=polypsDataset, ckpt_path="best")

    elif args.model_name == "sdnet_gradcam":
        trainer = Trainer(
            max_epochs=args.max_epochs,
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

    elif args.model_name == "sdnet_style_nonadv":
        trainer = Trainer(
            max_epochs=args.max_epochs,
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(
            SDNetStyleNonADVLight,
            datamodule=polypsDataset,
            # ckpt_path="/processing/v.corbetta/sdnet_miccai/results/sdnet_gradcam_no_centre_5_best/last-v3.ckpt",
        )
        trainer.test(SDNetStyleNonADVLight, datamodule=polypsDataset, ckpt_path="best")

    elif args.model_name == "sdnet_anatomy_dis":
        trainer = Trainer(
            max_epochs=args.max_epochs,
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(SDNetAnatomyDis, datamodule=polypsDataset)
        trainer.test(SDNetAnatomyDis, datamodule=polypsDataset, ckpt_path="best")

    elif args.model_name == "gan_sdnet":
        trainer = Trainer(
            max_epochs=args.max_epochs,
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(
            GANSDNetLight,
            datamodule=polypsDataset,
            # ckpt_path="/projects/shift_review/sdnet_miccai/projects/shift_review/sdnet_miccai/gan_sdnet_no_centre_2_retrain/last.ckpt",
        )
        trainer.test(GANSDNetLight, datamodule=polypsDataset, ckpt_path="best")

    elif args.model_name == "gan_sdnet_gradcam":
        trainer = Trainer(
            max_epochs=args.max_epochs,
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(
            GANSDNetGradCamLight,
            datamodule=polypsDataset,
            ckpt_path="/projects/SiH_disentanglement/results/gan_sdnet_gradcam_no_centre_5/last.ckpt",
        )
        trainer.test(GANSDNetGradCamLight, datamodule=polypsDataset, ckpt_path="best")

    elif args.model_name == "sihunet":
        trainer = Trainer(
            max_epochs=args.max_epochs,
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(SiHUNetLight, datamodule=polypsDataset)
        trainer.test(SiHUNetLight, datamodule=polypsDataset, ckpt_path="best")

    elif args.model_name == "polyp_classifier":
        trainer = Trainer(
            max_epochs=args.max_epochs,
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(PolypClassifierLight, datamodule=polypsDataset)
        trainer.test(PolypClassifierLight, datamodule=polypsDataset, ckpt_path="best")

    elif args.model_name == "gradcam_unet":
        trainer = Trainer(
            max_epochs=args.max_epochs,
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(GradCamUNetLight, datamodule=polypsDataset)
        trainer.test(GradCamUNetLight, datamodule=polypsDataset, ckpt_path="best")

    elif args.model_name == "polyp_detector":
        trainer = Trainer(
            max_epochs=args.max_epochs,
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_classifier],
        )
        trainer.fit(PolypDetectorLight, datamodule=polypsDataset)
        trainer.test(PolypDetectorLight, datamodule=polypsDataset, ckpt_path="best")

    elif args.model_name == "gradcam_deeplab":
        trainer = Trainer(
            max_epochs=args.max_epochs,
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(GradCamDeepLabLight, datamodule=polypsDataset)
        trainer.test(GradCamDeepLabLight, datamodule=polypsDataset, ckpt_path="best")

    elif args.model_name == "deeplab":
        trainer = Trainer(
            max_epochs=args.max_epochs,
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(DeepLabLight, datamodule=polypsDataset)
        trainer.test(DeepLabLight, datamodule=polypsDataset, ckpt_path="best")


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
        "--csv-saliency",
        dest="csv_saliency",
        type=str,
        help="Specify path to centre saliency maps if needed",
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
