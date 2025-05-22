import dask
import numpy as np
import math
from .hydrodata import HydroDataset

class DaskDataLoader:
    """
    A DataLoader that uses Dask for parallel batch loading from a given dataset.
    """

    def __init__(self, cfg: dict, dataset: HydroDataset = None, dask_dataframe=None, drop_last: bool = True):
        """
        Initializes the DaskDataLoader.

        Args:
            cfg (dict): Configuration dictionary containing parameters for the DataLoader.
            dataset (HydroDataset): The dataset to load data from.
            dask_dataframe: Optional Dask DataFrame for efficient batch loading.
            drop_last (bool): Set to True to drop the last incomplete batch,
                              if the dataset size is not divisible by the batch_size.
                              If False and the size of dataset is not divisible by
                              the batch_size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.dask_dataframe = dask_dataframe
        self.batch_size = cfg.get("batch_size", 1)
        self.shuffle = cfg.get("shuffle", True)
        self.num_workers = cfg.get("num_workers", 1)
        self.drop_last = drop_last
        self.indices = list(range(len(self.dataset))) if self.dataset else None

    def __iter__(self):
        if self.dask_dataframe is not None:
            # Use Dask DataFrame for batch loading
            ddf = self.dask_dataframe
            num_samples = len(ddf)
            num_batches = num_samples // self.batch_size
            if not self.drop_last and num_samples % self.batch_size != 0:
                num_batches += 1
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, num_samples)
                batch = ddf.iloc[start_idx:end_idx].compute()
                yield batch
            return

        if self.shuffle and self.indices is not None:
            np.random.shuffle(self.indices)

        if self.indices is not None:
            num_samples = len(self.indices)
            num_batches = num_samples // self.batch_size
            if not self.drop_last and num_samples % self.batch_size != 0:
                num_batches += 1

            batch_indices_list = [
                self.indices[i * self.batch_size : min((i + 1) * self.batch_size, num_samples)]
                for i in range(num_batches)
            ]

            if self.num_workers > 0:
                delayed_batches = [dask.delayed(self.dataset.__getitems__)(batch_indices)
                                for batch_indices in batch_indices_list]
                # Compute in parallel using threads, yielding in order
                for batch in dask.compute(*delayed_batches, scheduler='threads', num_workers=self.num_workers):
                    yield batch
            else:
                for batch_indices in batch_indices_list:
                    yield self.dataset.__getitems__(batch_indices)

    def __len__(self):
        if self.dask_dataframe is not None:
            num_samples = len(self.dask_dataframe)
        else:
            num_samples = len(self.dataset)
        if self.drop_last:
            return num_samples // self.batch_size
        else:
            return math.ceil(num_samples / self.batch_size)