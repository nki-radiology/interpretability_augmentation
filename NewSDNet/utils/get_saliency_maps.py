import torch
import numpy as np
from NewSDNet.dataset import PolypsDataModule
from NewSDNet.utils.centres_names import *
from NewSDNet.utils.csv_dicts import csv_dict
from random import random
import lightning.pytorch as pl


def main(args):
    ### Seed everything ###
    random.seed(args.seed)
    pl.seed_everything(args.seed)
