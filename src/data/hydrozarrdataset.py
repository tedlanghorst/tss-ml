import xarray as xr

from .hydrodata import HydroDataset


class HydroZarrDataset(HydroDataset):
    """
    Hybrid: loads/processes as in-memory, then dumps to Zarr and reloads for disk-backed access.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_d = None  # Will be opened lazily in each worker

    def _load_or_read_basin_data(self):
        # Use parent code to load/process in memory
        x_d = super()._load_or_read_basin_data()

        # Load or dump to zarr for disk-backed access in workers
        hash_str = super().get_data_hash()
        self.zarr_cache = self.cfg.get("data_dir") / "cache" / (hash_str + ".zarr")
        if not self.zarr_cache.exists():
            print("Saving dataset to cache.")
            x_d.to_zarr(self.zarr_cache, mode="w")

        return x_d

    def __getitems__(self, indices):
        """
        Override to load data from Zarr cache.
        """
        if self.x_d is None:
            self.x_d = xr.open_zarr(self.zarr_cache)

        # Use parent logic to get items
        return super().__getitems__(indices)
