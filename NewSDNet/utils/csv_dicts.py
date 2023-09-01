"""This file contains the paths to the csv files with the different splits and pre-generated
gradcam visualization for all the experiments:
each dictionary contains the path to 3 csv files, train, validation and in-distribution test sets, for a total of 
7 dictionaries, one for each different centre and an extra one where all the centres are used for training and validation.
"""
from pathlib import Path

PATH_TO_CSV_FOLDER = Path(__file__).resolve().parents[1] / "csv_files"
PATH_TO_SPLITS = PATH_TO_CSV_FOLDER / "splits"
PATH_TO_SALIENCY = PATH_TO_CSV_FOLDER / "saliency_maps"
PATH_TO_RESULTS = Path(__file__).resolve().parents[2] / "results"

### Path to csvs containing the splits depending on the out-dist centre ###
csv_dict_all_centres = {
    "train": PATH_TO_SPLITS / "all_centres_train_split",
    "val": PATH_TO_SPLITS / "all_centres_val_split",
    "test": PATH_TO_SPLITS / "all_centres_in_test_split",
}

csv_dict_no_centre_1 = {
    "train": PATH_TO_SPLITS / "no_centre_1_train_split",
    "val": PATH_TO_SPLITS / "no_centre_1_val_split",
    "test": PATH_TO_SPLITS / "no_centre_1_in_test_split",
}

csv_dict_no_centre_2 = {
    "train": PATH_TO_SPLITS / "no_centre_2_train_split",
    "val": PATH_TO_SPLITS / "no_centre_2_val_split",
    "test": PATH_TO_SPLITS / "no_centre_2_in_test_split",
}

csv_dict_no_centre_3 = {
    "train": PATH_TO_SPLITS / "no_centre_3_train_split",
    "val": PATH_TO_SPLITS / "no_centre_3_val_split",
    "test": PATH_TO_SPLITS / "no_centre_3_in_test_split",
}

csv_dict_no_centre_4 = {
    "train": PATH_TO_SPLITS / "no_centre_4_train_split",
    "val": PATH_TO_SPLITS / "no_centre_4_val_split",
    "test": PATH_TO_SPLITS / "no_centre_4_in_test_split",
}

csv_dict_no_centre_5 = {
    "train": PATH_TO_SPLITS / "no_centre_5_train_split",
    "val": PATH_TO_SPLITS / "no_centre_5_val_split",
    "test": PATH_TO_SPLITS / "no_centre_5_in_test_split",
}

csv_dict_no_centre_6 = {
    "train": PATH_TO_SPLITS / "no_centre_6_train_split",
    "val": PATH_TO_SPLITS / "no_centre_6_val_split",
    "test": PATH_TO_SPLITS / "no_centre_6_in_test_split",
}

# Map dict to corresponding centre, so that when out_centres is specified in the config file
# the right csv files are automatically selected
csv_split_mapping = {
    "no_out_dist": csv_dict_all_centres,
    "centre1": csv_dict_no_centre_1,
    "centre2": csv_dict_no_centre_2,
    "centre3": csv_dict_no_centre_3,
    "centre4": csv_dict_no_centre_4,
    "centre5": csv_dict_no_centre_5,
    "centre6": csv_dict_no_centre_6,
}

### Paths to pre-generated gradcam visualizations ###

csv_saliency_mapping = {
    "no_out_dist": PATH_TO_SALIENCY / "saliency_train_all_centres.csv",
    "centre1": PATH_TO_SALIENCY / "saliency_train_no_centre_1.csv",
    "centre2": PATH_TO_SALIENCY / "saliency_train_no_centre_2.csv",
    "centre3": PATH_TO_SALIENCY / "saliency_train_no_centre_3.csv",
    "centre4": PATH_TO_SALIENCY / "saliency_train_no_centre_4.csv",
    "centre5": PATH_TO_SALIENCY / "saliency_train_no_centre_5.csv",
    "centre6": PATH_TO_SALIENCY / "saliency_train_no_centre_6.csv",
}

### Names to giv to csv split files depending on the out-dist centre ###

csv_names_dict = {
    "no_out_dist": "all_centres",
    "centre1": "no_centre_1",
    "centre2": "no_centre_2",
    "centre3": "no_centre_3",
    "centre4": "no_centre_4",
    "centre5": "no_centre_5",
    "centre6": "no_centre_6",
}
