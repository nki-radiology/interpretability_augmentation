""" Class to define a LightningDataModule for the PolypGen2021 dataset
"""

from collections import defaultdict
import itertools
from pathlib import Path
from typing import Dict, List, Optional
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import numpy as np
import torch
import torch.nn as nn
from skimage import io
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
import albumentations as A
import pandas as pd
from sklearn.model_selection import train_test_split
from albumentations.pytorch.transforms import ToTensorV2
from NewSDNet.utils.paths_centres import *
from NewSDNet.utils.csv_dicts import PATH_TO_SPLITS
from NewSDNet.utils.rsynch import rsync_data
from NewSDNet.utils.patients import (
    patients_centre_six,
    patients_centre_five,
    patients_centre_two,
    patients_centre_three,
)


class PolypsDataset(Dataset):
    """Class to define a Pytorch Dataset for the PolypGen2021 dataset"""

    def __init__(
        self,
        images_paths: list[Path],
        labels_paths: list[Path],
        centre_lbl: list[int],
        load_saliency: bool = False,
        transform=None,
        saliency_paths=None,
    ):
        self.images_paths = images_paths
        self.labels_paths = labels_paths
        self.centre_lbl = centre_lbl
        self.load_saliency = load_saliency
        self.saliency_paths = saliency_paths
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image = io.imread(self.images_paths[idx])  # Tensor[C,H,W]::uint8 \in [0, 255].
        label = io.imread(self.labels_paths[idx])  # Tensor[H,W]::uint8 \in [0, 255].
        label = (np.where(label > 128, 1, 0)).astype(np.uint64)

        # We want to keep track from which centre the images come from, so that we can log
        # this info during training, validation and testing
        mapping = {"C1": 0, "C2": 1, "C3": 2, "C4": 3, "C5": 4, "C6": 5}
        for key in mapping.keys():
            if key in str(self.images_paths[idx]):
                centre = self.centre_lbl[mapping.get(key)]

        # To speed up training we have already computed the gradcam visualizations
        # if we are doing experiments with the interpretability-guided augmentation
        # we also need to load them
        if self.load_saliency:
            # Tensor[H,W]::uint64 \in [0,1]
            saliency_torch = torch.load(
                self.saliency_paths[idx], map_location=torch.device("cpu")
            )
            saliency_np = saliency_torch.squeeze(0).permute(1, 2, 0).numpy().squeeze(-1)

            data = self.transform(image=image, masks=[label, saliency_np])
            data["masks"][0] = data["masks"][0].type(torch.LongTensor)
            data["masks"][1] = data["masks"][1].type(torch.LongTensor)
            data["masks"][1] = data["masks"][1].repeat(
                8, 1, 1
            )  # for SDNet integration we create 8 channels of the gradcam

            return data["image"], data["masks"][0], centre, data["masks"][1]
        else:
            data = self.transform(image=image, mask=label)
            data["mask"] = data["mask"].type(torch.LongTensor)
            return (
                data["image"],
                data["mask"],
                centre,
            )  # , [self.images_paths[idx]]


class PolypsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        imgs_centres: dict[str, Path],
        seg_centres: dict[str, Path],
        imgs_out_test_centre: dict[str, Path],
        seg_out_test_centre: dict[str, Path],
        centre_lbl: list[int],
        save_path: Path,
        train_batch_size: int,
        num_workers: int,
        seed: int,
        from_csv: bool = False,
        per_patient: bool = True,
        csv_file_name: list[str] = None,
        path_to_csvs: dict[str, Path] = None,
        percentage_train: float = 0.8,
        load_saliency: bool = False,
        csv_saliency=None,
        flag_no_centres=False,
    ) -> None:
        super().__init__()
        self.flag_no_centres = flag_no_centres
        self.imgs_centres = imgs_centres
        self.seg_centres = seg_centres
        self.imgs_out_test_centre = imgs_out_test_centre
        self.seg_out_test_centre = seg_out_test_centre
        self.centre_lbl = centre_lbl
        self.csv_file_name = csv_file_name
        self.save_path = save_path
        self.train_batch_size = train_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.from_csv = from_csv
        self.per_patient = per_patient
        self.percentage_train = percentage_train
        self.path_to_csv_train = path_to_csvs["train"]
        self.path_to_csv_val = path_to_csvs["val"]
        self.path_to_csv_test = path_to_csvs["test"]
        self.load_saliency = load_saliency
        self.csv_saliency = csv_saliency
        self.saliency_maps = None
        self.train_imgs: list[torch.Tensor] = None
        self.train_lbls: list[torch.Tensor] = None
        self.val_imgs: list[torch.Tensor] = None
        self.val_lbls: list[torch.Tensor] = None
        self.in_test_imgs: list[torch.Tensor] = None
        self.in_test_lbls: list[torch.Tensor] = None
        self.out_test_imgs: list[torch.Tensor] = None
        self.out_test_lbls: list[torch.Tensor] = None
        self.train_transforms: A.transforms = None
        self.val_transforms: A.transforms = None
        self.train_dataset: Dataset = None
        self.val_dataset: Dataset = None
        self.in_test_dataset: Dataset = None
        self.out_test_dataset: Dataset = None

    def get_patients_per_centre(
        self,
        imgs_ctr: Dict[str, Path],
        gts_ctr: Dict[str, Path],
    ):
        """Function to group the frames coming from the same patient for
        each centre
        """
        imgs_dict = defaultdict(list)
        for centre_name, centre_path in imgs_ctr.items():
            if centre_name in ["centre1"]:
                # Centre1 has info of the patient in the frame name: e.g. 144OLC1_100H0022.jpg and 144OLCV1_100H0024.jpg
                imgs = sorted(centre_path.glob("*.jpg"))
                for impath in imgs:
                    imgs_dict[f"C1_{impath.stem[:3]}"].append(impath)

            elif centre_name in ["centre2"]:
                # For Centre2, Centre3, Centre5 and Centre6 the patients' IDs are in utils/patients.py
                imgs = sorted(
                    centre_path.glob("*.jpg"),
                    key=lambda path: int(path.stem.rsplit("_", 1)[1]),
                )

                for impath, patient_id in zip(imgs, patients_centre_two):
                    imgs_dict[f"C2_{patient_id}"].append(impath)

            elif centre_name in ["centre3"]:
                imgs = sorted(
                    centre_path.glob("*.jpg"),
                    key=lambda path: int(path.stem.rsplit("_", 2)[2]),
                )

                for impath, patient_id in zip(imgs, patients_centre_three):
                    imgs_dict[f"C3_{patient_id}"].append(impath)

            elif centre_name in ["centre4"]:
                # Centre4 has info of the patient in the frame name: e.g. 4_endocv2021_poitive_34.jpg and 4_endocv2021_positive_954.jpg
                imgs = sorted(centre_path.glob("*.jpg"), key=lambda path: path.stem)

                for impath in imgs:
                    imgs_dict[f"C4_{impath.stem[:2]}"].append(impath)

            elif centre_name in ["centre5"]:
                imgs = sorted(
                    centre_path.glob("*.jpg"),
                    key=lambda path: int(path.stem.rsplit("_", 2)[2]),
                )

                for impath, patient_id in zip(imgs, patients_centre_five):
                    imgs_dict[f"C5_{patient_id}"].append(impath)

            elif centre_name in ["centre6"]:
                imgs = sorted(
                    centre_path.glob("*.jpg"),
                    key=lambda path: int(path.stem.rsplit("_", 2)[2]),
                )

                for impath, patient_id in zip(imgs, patients_centre_six):
                    imgs_dict[f"C6_{patient_id}"].append(impath)

        gts_dict = defaultdict(list)
        for centre_name, centre_path in gts_ctr.items():
            if centre_name in ["centre1"]:
                # deduce from filename
                gts = sorted(centre_path.glob("*.jpg"))
                for gtpath in gts:
                    gts_dict[f"C1_{gtpath.stem[:3]}"].append(gtpath)

            elif centre_name in ["centre2"]:
                gts = sorted(
                    centre_path.glob("*.jpg"),
                    key=lambda path: int(path.stem.rsplit("_", 2)[1]),
                )

                for gtpath, patient_id in zip(gts, patients_centre_two):
                    gts_dict[f"C2_{patient_id}"].append(gtpath)

            elif centre_name in ["centre3"]:
                gts = sorted(
                    centre_path.glob("*.jpg"),
                    key=lambda path: int(path.stem.rsplit("_", 3)[2]),
                )

                for gtpath, patient_id in zip(gts, patients_centre_three):
                    gts_dict[f"C3_{patient_id}"].append(gtpath)

            elif centre_name in ["centre4"]:
                gts = sorted(centre_path.glob("*.jpg"), key=lambda path: path.stem)

                for gtpath in gts:
                    gts_dict[f"C4_{gtpath.stem[:2]}"].append(gtpath)

            elif centre_name in ["centre5"]:
                gts = sorted(
                    centre_path.glob("*.jpg"),
                    key=lambda path: int(path.stem.rsplit("_", 3)[2]),
                )

                for gtpath, patient_id in zip(gts, patients_centre_five):
                    gts_dict[f"C5_{patient_id}"].append(gtpath)

            elif centre_name in ["centre6"]:
                gts = sorted(
                    centre_path.glob("*.jpg"),
                    key=lambda path: int(path.stem.rsplit("_", 3)[2]),
                )

                for gtpath, patient_id in zip(gts, patients_centre_six):
                    gts_dict[f"C6_{patient_id}"].append(gtpath)

        return imgs_dict, gts_dict

    def get_list_imgs_gts(self, imgs_ctr, gts_ctr) -> tuple[list[Path, list[Path]]]:
        """Function to extract paths of images and ground truths given the centres' names,
        not taking into consideration the per patient split"""
        imgs_tot = []
        segs_tot = []

        for centre_name, centre_path in imgs_ctr.items():
            if centre_name in ["centre1"]:
                imgs = sorted(centre_path.glob("*.jpg"))
                imgs_tot.extend(imgs)
            elif centre_name in ["centre2"]:
                imgs = sorted(
                    centre_path.glob("*.jpg"),
                    key=lambda path: int(path.stem.rsplit("_", 1)[1]),
                )
                imgs_tot.extend(imgs)
            elif centre_name in ["centre3", "centre5", "centre6"]:
                imgs = sorted(
                    centre_path.glob("*.jpg"),
                    key=lambda path: int(path.stem.rsplit("_", 2)[2]),
                )

                imgs_tot.extend(imgs)
            elif centre_name in ["centre4"]:
                imgs = sorted(
                    centre_path.glob("*.jpg"),
                    key=lambda path: path.stem,
                )

                imgs_tot.extend(imgs)

        for centre_name, centre_path in gts_ctr.items():
            if centre_name in ["centre1"]:
                segs = sorted(centre_path.glob("*.jpg"))
                segs_tot.extend(segs)
            elif centre_name in ["centre2"]:
                segs = sorted(
                    centre_path.glob("*.jpg"),
                    key=lambda path: int(path.stem.rsplit("_", 2)[1]),
                )

                segs_tot.extend(segs)
            elif centre_name in ["centre3", "centre5", "centre6"]:
                segs = sorted(
                    centre_path.glob("*.jpg"),
                    key=lambda path: int(path.stem.rsplit("_", 3)[2]),
                )

                segs_tot.extend(segs)
            elif centre_name in ["centre4"]:
                segs = list(
                    sorted(
                        centre_path.glob("*.jpg"),
                        key=lambda path: path.stem[: -len("_mask")],
                    )
                )
                segs_tot.extend(segs)

        return imgs_tot, segs_tot

    def get_list_from_csv(self, csv_file) -> tuple[list[Path, list[Path]]]:
        """Function to get images and ground truths from .csv file, when these are available"""

        dataframe = pd.read_csv(csv_file)
        imgs_tot = dataframe["images"].tolist()
        segs_tot = dataframe["labels"].tolist()

        return imgs_tot, segs_tot

    def prepare_data(self) -> None:
        "Function to load images and labels depending whether csv files with the splits are available or not"

        # Load the paths where the gradcam visualizations are saved
        if self.load_saliency == True:
            self.saliency_maps = pd.read_csv(self.csv_saliency)["gradcam maps"].tolist()

        # If csv files for the splits are available, load them from csv files
        if self.from_csv == True:
            self.train_imgs, self.train_lbls = self.get_list_from_csv(
                self.path_to_csv_train
            )
            self.val_imgs, self.val_lbls = self.get_list_from_csv(self.path_to_csv_val)
            self.in_test_imgs, self.in_test_lbls = self.get_list_from_csv(
                self.path_to_csv_test
            )

        # If the csv files are not available and we want to perform the per patient split
        elif self.per_patient:
            patients_images, patients_gts = self.get_patients_per_centre(
                self.imgs_centres, self.seg_centres
            )
            patients_ids = list(patients_images.keys())
            patients_id_train, patients_no_train = train_test_split(
                patients_ids, train_size=self.percentage_train, random_state=self.seed
            )
            patients_id_val, patients_id_in_test = train_test_split(
                patients_no_train, train_size=0.5, random_state=self.seed
            )
            self.train_imgs = list(
                itertools.chain.from_iterable(
                    [patients_images[k] for k in patients_id_train]
                )
            )
            self.train_lbls = list(
                itertools.chain.from_iterable(
                    [patients_gts[k] for k in patients_id_train]
                )
            )
            self.val_imgs = list(
                itertools.chain.from_iterable(
                    [patients_images[k] for k in patients_id_val]
                )
            )
            self.val_lbls = list(
                itertools.chain.from_iterable(
                    [patients_gts[k] for k in patients_id_val]
                )
            )
            self.in_test_imgs = list(
                itertools.chain.from_iterable(
                    [patients_images[k] for k in patients_id_in_test]
                )
            )
            self.in_test_lbls = list(
                itertools.chain.from_iterable(
                    [patients_gts[k] for k in patients_id_in_test]
                )
            )
            train_df = pd.DataFrame(
                list(zip(self.train_imgs, self.train_lbls)),
                columns=["images", "labels"],
            )
            csv_name_train = self.csv_file_name + "_train_split"
            train_df.to_csv(PATH_TO_SPLITS / csv_name_train)
            val_df = pd.DataFrame(
                list(zip(self.val_imgs, self.val_lbls)), columns=["images", "labels"]
            )
            csv_name_val = self.csv_file_name + "_val_split"
            val_df.to_csv(PATH_TO_SPLITS / csv_name_val)
            in_test_df = pd.DataFrame(
                list(zip(self.in_test_imgs, self.in_test_lbls)),
                columns=["images", "labels"],
            )
            csv_name_test = self.csv_file_name + "_in_test_split"
            in_test_df.to_csv(PATH_TO_SPLITS / csv_name_test)

        else:
            imgs_paths, segs_paths = self.get_list_imgs_gts(
                self.imgs_centres, self.seg_centres
            )
            (
                self.train_imgs,
                no_train_imgs,
                self.train_lbls,
                no_train_lbls,
            ) = train_test_split(
                imgs_paths,
                segs_paths,
                train_size=self.percentage_train,
                random_state=self.seed,
            )
            train_df = pd.DataFrame(
                list(zip(self.train_imgs, self.train_lbls)),
                columns=["images", "labels"],
            )
            csv_name_train = self.csv_file_name + "_train_split"
            train_df.to_csv(PATH_TO_SPLITS / csv_name_train)

            (
                self.val_imgs,
                self.in_test_imgs,
                self.val_lbls,
                self.in_test_lbls,
            ) = train_test_split(
                no_train_imgs, no_train_lbls, train_size=0.5, random_state=self.seed
            )

            val_df = pd.DataFrame(
                list(zip(self.val_imgs, self.val_lbls)), columns=["images", "labels"]
            )
            csv_name_val = self.csv_file_name + "_val_split"
            val_df.to_csv(PATH_TO_SPLITS / csv_name_val)
            in_test_df = pd.DataFrame(
                list(zip(self.in_test_imgs, self.in_test_lbls)),
                columns=["images", "labels"],
            )
            csv_name_test = self.csv_file_name + "_in_test_split"
            in_test_df.to_csv(PATH_TO_SPLITS / csv_name_test)

        if not self.flag_no_centres:
            self.out_test_imgs, self.out_test_lbls = self.get_list_imgs_gts(
                self.imgs_out_test_centre, self.seg_out_test_centre
            )

    def get_train_transforms(self) -> A.transforms:
        """Function that returns transforms and augmentations to apply on training data"""
        train_transforms = A.Compose(  # colorjitter improved performance both for sdnet and unet but I don't think it makes sense for the style-domain discr
            [
                A.Resize(512, 512),
                A.Normalize(),
                A.HorizontalFlip(always_apply=False, p=0.5),
                A.VerticalFlip(always_apply=False, p=0.5),
                A.Rotate(),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.2,
                    always_apply=False,
                    p=0.3,
                ),
                # A.GaussNoise(per_channel=True, always_apply=False, p=0.3),
                # A.RandomBrightnessContrast(always_apply=False, p=0.3),
                ToTensorV2(),
            ]
        )
        return train_transforms

    def get_val_transforms(self) -> A.transforms:
        """Function that returns transforms to apply to validation/testing data"""
        val_transforms = A.Compose([A.Resize(512, 512), A.Normalize(), ToTensorV2()])
        return val_transforms

    def setup(self, stage: str = None) -> None:
        """Function to get pytorch datasets for train, validationand in/out testing"""
        self.train_transforms = self.get_train_transforms()
        self.val_transforms = self.get_val_transforms()
        if self.load_saliency:
            self.train_dataset = PolypsDataset(
                self.train_imgs,
                self.train_lbls,
                self.centre_lbl,
                load_saliency=self.load_saliency,
                transform=self.train_transforms,
                saliency_paths=self.saliency_maps,
            )
        else:
            self.train_dataset = PolypsDataset(
                self.train_imgs,
                self.train_lbls,
                self.centre_lbl,
                load_saliency=False,
                transform=self.train_transforms,
            )
        self.val_dataset = PolypsDataset(
            self.val_imgs,
            self.val_lbls,
            self.centre_lbl,
            load_saliency=False,
            transform=self.val_transforms,
        )

        self.in_test_dataset = PolypsDataset(
            self.in_test_imgs,
            self.in_test_lbls,
            self.centre_lbl,
            load_saliency=False,
            transform=self.val_transforms,
        )

        if not self.flag_no_centres:
            self.out_test_dataset = PolypsDataset(
                self.out_test_imgs,
                self.out_test_lbls,
                self.centre_lbl,
                load_saliency=False,
                transform=self.val_transforms,
            )

    def train_dataloader(self) -> DataLoader:
        """Function to get train dataloader"""
        return DataLoader(
            self.train_dataset,
            self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        "Function to get validation dataloader"
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Function to get in-distribution test dataloader"""

        in_dist_dl = DataLoader(
            self.in_test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        if not self.flag_no_centres:
            out_dist_dl = DataLoader(
                self.out_test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            return [in_dist_dl, out_dist_dl]
        else:
            return in_dist_dl
