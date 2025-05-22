from .hydrodata import HydroDataset
from torch.utils.data import DataLoader, get_worker_info


def collate_fn(sample):
    return sample

def worker_init_fn(worker_id):
    print(worker_id)
    worker_info = get_worker_info()
    worker_info.dataset.load_shard(chunk_idx=worker_id)


class ShardedHydroDataset:
    def __init__(self, cfg: dict, master_scaling: dict, master_len: int, n_chunks: int):
        # Only store config and normalization info, do not load data yet
        self.cfg = cfg
        self.scaling = master_scaling
        self.dataset = None  # Will hold HydroDataset instance after loading shard
        self._reported_shard_len = master_len // n_chunks
        self.cfg["n_chunks"] = n_chunks
        self.data_loaded = False

    def load_shard(self, chunk_idx: int):
        if self.data_loaded:
            print(f"Data already loaded for worker n. {chunk_idx}")
            return 
        self.data_loaded = True

        # Set chunking in config
        self.cfg["chunk_idx"] = chunk_idx
        self.cfg["num_dataset_workers"] = 0
        self.cfg['use_cache'] = False
        # Create HydroDataset instance for this shard
        self.dataset = HydroDataset(self.cfg, scaling=self.scaling)

    def __getitems__(self, ids: list[int]):
        if not self.data_loaded:
            print("Data not loaded yet!")

        return self.dataset.__getitems__(ids)

    def __len__(self):
        if self.dataset:
            return len(self.dataset)
        else:
            return self._reported_shard_len

