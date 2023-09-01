import json
from NewSDNet.dataset import PolypsDataModule
from NewSDNet.utils.centres_names import *
from NewSDNet.utils.csv_dicts import (
    csv_split_mapping,
    csv_saliency_mapping,
    csv_names_dict,
    PATH_TO_RESULTS,
)
from NewSDNet.utils.centre_labels import centres_labels
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

    seed = args["train"]["seed"]
    ### Seed everything ###
    random.seed(seed)
    pl.seed_everything(seed)

    ### Initialize logging ###
    wandb_logger = WandbLogger(
        name=args["wandb"]["wandb_name"],
        project=args["wandb"]["project_name"],
    )
    save_path = args["train"]["save_path"]
    checkpoint_callback = ModelCheckpoint(
        dirpath=PATH_TO_RESULTS / save_path,
        save_top_k=1,
        save_last=True,
        monitor="val_dice",
        mode="max",
    )
    checkpoint_classifier = ModelCheckpoint(
        dirpath=PATH_TO_RESULTS / save_path,
        save_top_k=1,
        save_last=True,
        monitor="Validation accuracy",
        mode="max",
    )

    # resolve out distribution test centre
    out_centres = args["defaults"]["dataset"]["out_centres"]
    load_saliency = args["defaults"]["dataset"]["load_saliency"]
    imgs_test_centre = [
        {center_string: imgs_centres.pop(center_string)}
        for center_string in out_centres
    ]
    seg_test_centre = [
        {center_string: seg_centres.pop(center_string)} for center_string in out_centres
    ]

    if out_centres == []:
        csv_dict = csv_split_mapping["no_out_dist"]
    else:
        csv_dict = csv_split_mapping[out_centres]

    if out_centres == []:
        csv_dict = csv_split_mapping["no_out_dist"]
    else:
        csv_dict = csv_split_mapping[out_centres]

    if load_saliency == True and out_centres == []:
        csv_saliency = csv_saliency_mapping["no_out_dist"]
    elif load_saliency == True and out_centres != []:
        csv_saliency = csv_saliency_mapping[out_centres]
    elif load_saliency == False:
        csv_saliency = None

    if out_centres == []:
        csv_file_name = csv_names_dict["no_out_dist"]
    else:
        centre_lbl = csv_names_dict[out_centres]

    ### Initialize LightningDataModule ###

    polypsDataset = PolypsDataModule(
        imgs_centres=imgs_centres,
        seg_centres=seg_centres,
        imgs_out_test_centre=imgs_test_centre,
        seg_out_test_centre=seg_test_centre,
        centre_lbl=centre_lbl,
        csv_file_name=csv_file_name,
        save_path=PATH_TO_RESULTS / save_path,
        train_batch_size=args["train"]["train_batch_size"],
        from_csv=args["defaults"]["dataset"]["from_csv"],
        per_patient=args["defaults"]["dataset"]["per_patient"],
        load_saliency=args["defaults"]["dataset"]["load_saliency"],
        csv_saliency=csv_saliency,
        num_workers=args["train"]["num_workers"],
        seed=seed,
        path_to_csvs=csv_dict,
        percentage_train=args["train"]["percentage_train"],
        flag_no_centres=out_centres == [],
    )

    ### Initialize model ###

    if args["train"]["model_name"] == "sdnet_gradcam":
        model = SDNet(**args["sdnet"]["model_params"])
        SDNetGradCamLight = SDNetGradCamLightning(
            model=model,
            lr=args["train"]["learning_rate"],
            img_logger=wandb_logger,
            batch_size=args["defaults"]["dataset"]["train_batch_size"],
            **args["sdnet"]["lightning"]
        )

    if args["train"]["model_name"] == "classifier":
        ResNetLight = ResNetLightning(
            lr=args["train"]["learning_rate"],
            img_logger=wandb_logger,
            batch_size=args["defaults"]["dataset"]["train_batch_size"],
            **args["classifier"]
        )

    if args["train"]["model_name"] == "sdnet":
        model = SDNet(**args["sdnet"]["model_params"])
        SDNetLight = SDNetLightning(
            model=model,
            lr=args["train"]["learning_rate"],
            img_logger=wandb_logger,
            batch_size=args["defaults"]["dataset"]["train_batch_size"],
            **args["sdnet"]["lightning"]
        )

    if args["train"]["model_name"] == "unet":
        model = UNet(**args["unet"])
        UNetLight = UNetLightning(
            model=model,
            lr=args["train"]["learning_rate"],
            img_logger=wandb_logger,
            save_path=PATH_TO_RESULTS / save_path,
        )

    if args["train"]["model_name"] == "gradcam_unet":
        model = UNet(**args["unet"])
        GradCamUNetLight = GradCamUNetLightning(
            model=model,
            lr=args["train"]["learning_rate"],
            img_logger=wandb_logger,
            batch_size=args["defaults"]["dataset"]["train_batch_size"],
            save_path=PATH_TO_RESULTS / save_path,
        )
    if args["train"]["model_name"] == "gradcam_deeplab":
        model_map = {"deeplabv3plus_resnet101": deeplabv3plus_resnet101}
        model = model_map["deeplabv3plus_resnet101"](
            num_classes=2, output_stride=16, pretrained_backbone=False
        )
        GradCamDeepLabLight = GradCamDeepLabLightning(
            model=model,
            lr=args["train"]["learning_rate"],
            img_logger=wandb_logger,
            batch_size=args["defaults"]["dataset"]["train_batch_size"],
            save_path=PATH_TO_RESULTS / save_path,
        )
    if args["train"]["model_name"] == "deeplab":
        model_map = {"deeplabv3plus_resnet101": deeplabv3plus_resnet101}
        model = model_map["deeplabv3plus_resnet101"](
            num_classes=2, output_stride=16, pretrained_backbone=False
        )
        DeepLabLight = DeepLabLightning(
            model=model,
            lr=args["train"]["learning_rate"],
            img_logger=wandb_logger,
            batch_size=args["defaults"]["dataset"]["train_batch_size"],
            save_path=PATH_TO_RESULTS / save_path,
        )

    ### Training ###

    if args["train"]["model_name"] == "sdnet":
        trainer = Trainer(
            max_epochs=args["train"]["max_epochs"],
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )

        trainer.fit(
            SDNetLight,
            datamodule=polypsDataset,
        )
        trainer.test(SDNetLight, datamodule=polypsDataset, ckpt_path="best")

    elif args["train"]["model_name"] == "classifier":
        trainer = Trainer(
            max_epochs=args["train"]["max_epochs"],
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_classifier],
            inference_mode=False,
        )

        trainer.fit(
            ResNetLight,
            datamodule=polypsDataset,
        )
        trainer.test(ResNetLight, datamodule=polypsDataset)

    elif args["train"]["model_name"] == "sdnet_gradcam":
        trainer = Trainer(
            max_epochs=args["train"]["max_epochs"],
            num_sanity_val_steps=0,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(
            SDNetGradCamLight,
            datamodule=polypsDataset,
        )
        trainer.test(SDNetGradCamLight, datamodule=polypsDataset, ckpt_path="best")

    elif args["train"]["model_name"] == "unet":
        trainer.fit(
            UNetLight,
            datamodule=polypsDataset,
        )
        trainer.test(UNetLight, datamodule=polypsDataset, ckpt_path="best")

    elif args["train"]["model_name"] == "gradcam_unet":
        trainer = Trainer(
            max_epochs=args["train"]["max_epochs"],
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(GradCamUNetLight, datamodule=polypsDataset)
        trainer.test(GradCamUNetLight, datamodule=polypsDataset, ckpt_path="best")

    elif args["train"]["model_name"] == "gradcam_deeplab":
        trainer = Trainer(
            max_epochs=args["train"]["max_epochs"],
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(GradCamDeepLabLight, datamodule=polypsDataset)
        trainer.test(GradCamDeepLabLight, datamodule=polypsDataset, ckpt_path="best")

    elif args["train"]["model_name"] == "deeplab":
        trainer = Trainer(
            max_epochs=args["train"]["max_epochs"],
            num_sanity_val_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(DeepLabLight, datamodule=polypsDataset)
        trainer.test(DeepLabLight, datamodule=polypsDataset, ckpt_path="best")


if __name__ == "__main__":
    main()
