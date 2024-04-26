from argparse import Namespace
from typing import Optional
import math
import os
from os.path import dirname, abspath
import yaml

import numpy as np
import xarray as xr
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.xarray_dataset import XarrayDataset

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
ON_GPU = config['ON_GPU']
NUM_AVAIL_CPUS = len(os.sched_getaffinity(0))
NUM_AVAIL_GPUS = torch.cuda.device_count()
TRANSFORM = config['TRANSFORM']
BATCH_SIZE = config['BATCH_SIZE']

# sensible multiprocessing defaults: at most one worker per CPU
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS
# but in distributed data parallel mode, we launch a training on each GPU
# so must divide out to keep total at one worker per CPU
DEFAULT_NUM_WORKERS = (
    NUM_AVAIL_CPUS // NUM_AVAIL_GPUS if NUM_AVAIL_GPUS else DEFAULT_NUM_WORKERS
)
N_CLASS = config['N_CLASS']
N_BAND = config['N_BAND']

PROCESSED_DATA_DIRNAME = config['PROCESSED_DATA_DIRNAME']
PROCESSED_TRAIN_DATA_FILENAME = config['PROCESSED_TRAIN_DATA_FILENAME']
PROCESSED_TEST_DATA_FILENAME = config['PROCESSED_TEST_DATA_FILENAME']
PROCESSED_VALID_DATA_FILENAME = config['PROCESSED_VALID_DATA_FILENAME']


class XarrayDataModule(pl.LightningDataModule):
    """lightning data module for xarray data"""

    def __init__(self) -> None:
        super().__init__()
        
        self.num_workers = DEFAULT_NUM_WORKERS
        self.transform = TRANSFORM
        self.batch_size = BATCH_SIZE
        self.on_gpu = ON_GPU
        
        self.num_classes = N_CLASS
        self.num_bands = N_BAND

        self.data_train = None
        self.data_test = None
        self.data_val = None

    def prepare_data(self, *args, **kwargs) -> None:
        """download data here"""

    def setup(self, stage: Optional[str] = None):
        """
        Read downloaded data
        Setup Datasets
        Split the dataset into train/val/test."""

        # create paths and filenames
        parent = dirname(abspath("__file__"))
        self.full_path = parent + PROCESSED_DATA_DIRNAME

        self.full_test_file = self.full_path + "/" + PROCESSED_TEST_DATA_FILENAME
        self.full_valid_file = self.full_path + "/" + PROCESSED_VALID_DATA_FILENAME
        self.full_train_file = self.full_path + "/" + PROCESSED_TRAIN_DATA_FILENAME

        # load data
        try:
            traindata = xr.open_zarr(self.full_train_file)
        except FileNotFoundError:
            print(f"Train data file {self.full_train_file} not found")

        try:
            validdata = xr.open_zarr(self.full_valid_file)
        except FileNotFoundError:
            print(f"Valid data file {self.full_valid_file} not found")

        try:
            testdata = xr.open_zarr(self.full_test_file)
        except FileNotFoundError:
            print(f"Test data file {self.full_test_file} not found")

        testind = list(range(0, len(testdata.batch)))
        testdata = testdata.reindex({'batch': testdata.batch})
        trainind = list(range(0, len(traindata.batch)))
        traindata = traindata.reindex({'batch': traindata.batch})
        validind = list(range(0, len(validdata.batch)))
        validdata = validdata.reindex({'batch': validdata.batch})

        self.data_train = XarrayDataset(traindata)
        self.data_test = XarrayDataset(testdata)
        self.data_val = XarrayDataset(validdata)
        
    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def __str__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "xarray example dataset\n"
            f"Num classes: {self.num_classes}\n"
            f"Num bands: {self.num_bands}\n"
        )
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))

        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min().item(), x.mean().item(), x.std().item(), x.max().item())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min().item(), y.max().item())}\n"
        )
        return basic + data
