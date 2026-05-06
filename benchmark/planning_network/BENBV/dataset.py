from collections import OrderedDict
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ObjViewBenbvNpzDataset(Dataset):
    """Reads ObjViewBench BENBV rollout npz files.

    Expected layout:
        data_root/<uid>/start000.npz
        data_root/<uid>/start001.npz

    Each npz stores arrays:
        P: [T, 4096, 6]
        S: [T, 20, 6]
        C: [T, 21, 1]
        y: [T, 20, 1]
    """

    def __init__(
        self,
        data_root,
        max_files=None,
        max_steps_per_file=None,
        cache_mode="lazy",
        cache_size=8,
    ):
        self.data_root = Path(data_root)
        self.max_steps_per_file = max_steps_per_file
        self.cache_mode = cache_mode
        self.cache_size = max(1, int(cache_size))
        self._cache = OrderedDict()

        files = sorted(self.data_root.glob("*/start*.npz"))
        if max_files is not None and max_files > 0:
            files = files[:max_files]
        if not files:
            raise RuntimeError(f"No start*.npz files found under: {self.data_root}")

        self.files = [str(p) for p in files]
        self.index = []
        self.arrays = {}
        self.file_steps = []

        for file_id, path in enumerate(self.files):
            with np.load(path) as z:
                steps = int(z["P"].shape[0])
            if max_steps_per_file is not None and max_steps_per_file > 0:
                steps = min(steps, int(max_steps_per_file))
            self.file_steps.append(steps)
            for step in range(steps):
                self.index.append((file_id, step))

        if self.cache_mode == "memory":
            for path in self.files:
                self.arrays[path] = self._read_npz(path)

        print(f"ObjView BENBV npz files: {len(self.files)}")
        print(f"ObjView BENBV samples: {len(self.index)}")

    def get_stats(self):
        pairs_by_uid = defaultdict(int)
        starts_by_uid = defaultdict(int)
        for path, steps in zip(self.files, self.file_steps):
            uid = Path(path).parent.name
            pairs_by_uid[uid] += int(steps)
            starts_by_uid[uid] += 1

        pairs = np.array(list(pairs_by_uid.values()), dtype=np.float32)
        starts = np.array(list(starts_by_uid.values()), dtype=np.float32)
        file_steps = np.array(self.file_steps, dtype=np.float32)

        return {
            "num_objects": int(len(pairs_by_uid)),
            "num_npz_files": int(len(self.files)),
            "num_samples": int(len(self.index)),
            "mean_starts_per_object": float(starts.mean()) if len(starts) else 0.0,
            "mean_steps_per_start": float(file_steps.mean()) if len(file_steps) else 0.0,
            "median_steps_per_start": float(np.median(file_steps)) if len(file_steps) else 0.0,
            "mean_pairs_per_object": float(pairs.mean()) if len(pairs) else 0.0,
            "median_pairs_per_object": float(np.median(pairs)) if len(pairs) else 0.0,
            "min_pairs_per_object": int(pairs.min()) if len(pairs) else 0,
            "max_pairs_per_object": int(pairs.max()) if len(pairs) else 0,
        }

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _read_npz(path):
        with np.load(path) as z:
            return {
                "P": z["P"].astype(np.float32),
                "S": z["S"].astype(np.float32),
                "C": z["C"].astype(np.float32),
                "y": z["y"].astype(np.float32),
            }

    def _get_arrays(self, path):
        if self.cache_mode == "memory":
            return self.arrays[path]

        if path in self._cache:
            arrays = self._cache.pop(path)
            self._cache[path] = arrays
            return arrays

        arrays = self._read_npz(path)
        self._cache[path] = arrays
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return arrays

    def __getitem__(self, idx):
        file_id, step = self.index[idx]
        arrays = self._get_arrays(self.files[file_id])
        return (
            torch.from_numpy(arrays["P"][step]).float(),
            torch.from_numpy(arrays["S"][step]).float(),
            torch.from_numpy(arrays["C"][step]).float(),
            torch.from_numpy(arrays["y"][step]).float(),
        )
