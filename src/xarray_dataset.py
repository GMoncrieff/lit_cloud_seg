from typing import Any, Callable, Optional, Tuple
import yaml
import einops

import torch
from torch.utils.data import Dataset
import xarray as xr

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
PREDICTOR_VAR = config['PREDICTOR_VAR']
LABEL_VAR = config['LABEL_VAR']

class XarrayDataset(Dataset):
    def __init__(
        self,
        xrdata: xr.Dataset,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        PyTorch Dataset adapter for xarray
        Parameters
        ----------
        Xarray : xr.Dataset
            An xarray dataset with a predictor and label variable. must have an indexed batch dim named batch
        transform : callable, optional
            A function/transform that takes in an array and returns a transformed version.
        target_transform : callable, optional
            A function/transform that takes in the target and transforms it.
        """

        self.xrdata = xrdata
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.xrdata.batch)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        xr_batch = self.xrdata.isel(batch=idx).load()

        x_batch = torch.tensor(data=xr_batch[PREDICTOR_VAR].data)
        y_batch = torch.tensor(data=xr_batch[LABEL_VAR].data).type(torch.int64)

        if self.transform:
            x_batch = self.transform(x_batch)

        if self.target_transform:
            y_batch = self.target_transform(y_batch)

        return x_batch, y_batch
