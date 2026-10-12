"""WheeledLab FKD training samples over OptCar's recorded processed shards.

Selection uses the shared processed loader. Sample slicing below is copied from
WheeledLab's FKD dataloader; stored windows may include one extra future state.
"""
import json
import operator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from tqdm import tqdm

from optcar.dataloader_config import DataloaderConfig
from optcar.dataset import DataConfig, FKDDataset as ProcessedDataset
from optcar.gui.dataset_viewer import read_dataset_config


class FKDDataset(Dataset):
    """Reference FKD H/F convention, with recorded train/test directories."""
    def __init__(self, dl_config: DataloaderConfig, future_horizon_steps: int,
                 history_horizon_steps: int, local_hmap_bf_size_px: Optional[int] = None,
                 local_hmap_bf_res: Optional[float] = None,
                 hmap_img_size: Optional[int] = None, no_map: bool = True,
                 is_test_set: bool = False, preload_to_ram: bool = False):
        if not isinstance(dl_config, DataloaderConfig):
            raise ValueError("Expected DataloaderConfig")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in
               (history_horizon_steps, future_horizon_steps)) or history_horizon_steps < 2 or future_horizon_steps < 2:
            raise ValueError("FKD history and future horizons must be integers >= 2")
        if dl_config.is_cmd_vel_in_data or dl_config.include_dbm_params:
            raise ValueError("OptCar processed datasets use the standard 42-column FKD layout")
        self.dl_config = dl_config
        device = dl_config.device
        self.device = (device if isinstance(device, torch.device) else torch.device(device)
                       if isinstance(device, str) else torch.device(f"cuda:{device}")
                       if isinstance(device, int) else torch.device("cpu"))
        self.dtype = dl_config.dtype
        self.future_horizon_steps = future_horizon_steps
        self.history_horizon_steps = history_horizon_steps
        self.total_horizon_len = history_horizon_steps + future_horizon_steps - 1
        self.local_hmap_bf_size_px = local_hmap_bf_size_px if local_hmap_bf_size_px is not None else 500
        self.local_hmap_bf_res = local_hmap_bf_res if local_hmap_bf_res is not None else .01
        self.hmap_img_size = hmap_img_size if hmap_img_size is not None else 1000
        self.no_map, self.is_test_set, self.is_val_set = no_map, is_test_set, False
        self.preload_to_ram = preload_to_ram
        self._dbm_dim = 0
        self._dbm_mean = self._dbm_std = self._dbm_is_constant = None
        paths = dl_config.test_dataset_path if is_test_set else dl_config.dataset_path
        if isinstance(paths, str):
            paths = [paths]
        if not isinstance(paths, list) or not paths or not all(isinstance(path, str) and path.strip() for path in paths):
            raise ValueError("dataset_path must be a string or list of strings")
        self.dataset_paths = [Path(path).expanduser().resolve() for path in paths]
        self.shards, self.dataset_file_mapping, self.shard_splits = [], [], []
        self._global_indices, self.selection_summary = None, []
        seen = set()
        selected_split = "test" if is_test_set else "train"
        for dataset_idx, root in enumerate(self.dataset_paths):
            for dataset in self._processed_datasets(root):
                for shard, split in zip(dataset.shards, dataset.shard_splits):
                    if split not in (None, selected_split) or shard[0] in seen:
                        continue
                    seen.add(shard[0])
                    self.shards.append(shard)
                    self.dataset_file_mapping.append(dataset_idx)
                    self.shard_splits.append(split)
        if not self.shards:
            raise ValueError(f"No {selected_split} windows found in {self.dataset_paths}")
        self.ends = np.cumsum([shard[1][0] for shard in self.shards]).tolist()
        # Physical bins remain float32; WheeledLab converts samples to dtype.
        ProcessedDataset._select_dataloader_trajectories(self, replace(dl_config, dtype=torch.float32), is_test_set)
        self.files = [str(shard[0]) for shard in self.shards]
        self.shapes = [shard[1] for shard in self.shards]
        self.file_traj_counts = [shape[0] for shape in self.shapes]
        self.cumulative_traj_counts = [0, *self.ends[:-1]]
        self.total_trajectories = self.ends[-1]
        if is_test_set and any(split is None for split in self.shard_splits):
            main_paths = dl_config.dataset_path
            main_paths = [main_paths] if isinstance(main_paths, str) else main_paths or []
            main_files = set()
            for path in main_paths:
                root = Path(path).expanduser().resolve()
                folder = root / "data" if (root / "data").is_dir() else root
                main_files.update(file.resolve() for file in folder.glob("*.bin"))
            if any(shard[0] in main_files for shard, split in zip(self.shards, self.shard_splits) if split is None):
                raise ValueError("Training and test dataset paths overlap; choose separate test bins")
        self._mmaps = [None] * len(self.files)
        if preload_to_ram:
            self._data_arrays = [None] * len(self.files)

            def load_single_file(args):
                idx, file, shape = args
                return idx, np.fromfile(file, dtype=np.float32, count=int(np.prod(shape))).reshape(shape)

            load_args = [(i, file, self.shapes[i]) for i, file in enumerate(self.files)]
            # WheeledLab's preload uses 16 threads.
            with ThreadPoolExecutor(max_workers=16) as executor:
                for idx, data in tqdm(executor.map(load_single_file, load_args),
                                      total=len(self.files), desc="Preloading dataset to RAM"):
                    self._data_arrays[idx] = data
        else:
            for i, file in enumerate(self.files):
                self._mmaps[i] = np.memmap(file, dtype=np.float32, mode="c", shape=self.shapes[i])
        dummy_shape = (1, 1) if no_map else (self.hmap_img_size, self.hmap_img_size)
        self.dummy_hmap_torch = torch.zeros(dummy_shape, dtype=self.dtype, device="cpu")

    def _processed_datasets(self, root):
        """Use saved physical dimensions before selecting the model horizon."""
        if not root.is_dir():
            raise ValueError(f"Dataset directory does not exist: {root}")
        manifests = [root / "manifest.json"] if (root / "manifest.json").is_file() else sorted(root.rglob("manifest.json"))
        if not manifests:
            for parent in root.parents:
                path = parent / "manifest.json"
                if path.is_file():
                    doc = json.loads(path.read_text())
                    if doc.get("schema", {}).get("features") == 42 and "episodes" in doc:
                        manifests = [path]
                        break
        if manifests:
            for manifest in manifests:
                doc = json.loads(manifest.read_text())
                if not doc.get("process_windows", True):
                    continue
                if "config" not in doc or "episodes" not in doc:
                    continue
                config = DataConfig(**{key: doc["config"][key] for key in DataConfig.__dataclass_fields__ if key in doc["config"]})
                self._check_physical_config(config, manifest)
                folder = manifest.parent if manifest.parent.is_relative_to(root) else root
                yield ProcessedDataset(folder, config, "all", allow_wheeledlab=True)
            return
        metadata = None
        for folder in (root, *root.parents):
            for relative in ("metadata/dataset_config.py", "metadata/schema.json", "schema.json"):
                candidate = folder / relative
                if candidate.is_file():
                    metadata = candidate
                    break
            if metadata:
                break
        config = (read_dataset_config(metadata) if metadata else
                  DataConfig(history_steps=self.history_horizon_steps,
                             future_steps=self.future_horizon_steps - 1, dt=self.dl_config.sim_dt))
        self._check_physical_config(config, metadata or root)
        yield ProcessedDataset(root, config, "all", allow_wheeledlab=True)

    def _check_physical_config(self, config, path):
        if config.history_steps != self.history_horizon_steps:
            raise ValueError(f"{path}: history differs from the model's body-frame anchor")
        if config.window_steps < self.total_horizon_len:
            raise ValueError(f"{path}: stored window is shorter than history + future - 1")
        if config.dt != self.dl_config.sim_dt:
            raise ValueError(f"{path}: timestep differs from dataloader sim_dt")

    @property
    def dbm_mean(self):
        return self._dbm_mean

    @property
    def dbm_std(self):
        return self._dbm_std

    @property
    def dbm_is_constant(self):
        return self._dbm_is_constant

    def __len__(self):
        return len(self._global_indices) if self._global_indices is not None else self.total_trajectories

    def _find_file_and_agent_idx(self, global_idx: int):
        """Find which file and local agent index corresponds to the global trajectory index."""
        file_idx = 0
        for _file_idx in range(len(self.cumulative_traj_counts)):
            if global_idx >= self.cumulative_traj_counts[_file_idx]:
                file_idx = _file_idx
            else:
                break
        
        agent_idx = global_idx - self.cumulative_traj_counts[file_idx]
        return file_idx, agent_idx

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single trajectory sample."""
        idx = operator.index(idx)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        
        if self._global_indices is not None:
            idx = self._global_indices[idx]
        file_idx, agent_idx = self._find_file_and_agent_idx(idx)

        if self.preload_to_ram:
            sample_np = self._data_arrays[file_idx][agent_idx]
        else:
            # Load data using memory mapping with the correct shape for this file
            memmap_shape = self.shapes[file_idx]

            # Lazily open memmap per worker process
            memmap_file = self._mmaps[file_idx]
            if memmap_file is None:
                memmap_file = np.memmap(
                    self.files[file_idx],
                    dtype=np.float32,
                    mode="c",
                    shape=memmap_shape,
                )
                self._mmaps[file_idx] = memmap_file
                print("Opened memmap file:", self.files[file_idx], "Total memmaps:", len(self._mmaps))
            
            sample_np = memmap_file[agent_idx]
        # if not sample_np.flags.writeable:
        #     sample_np = np.array(sample_np, copy=True)  # avoid non-writable warnings
        data = torch.from_numpy(sample_np).to(self.dtype)
        # OptCar stores the extra transition; FKD shares current between H/F.
        data = data[:self.total_horizon_len]
        # if self.device.type != "cpu":
        #     data = data.to(self.device, non_blocking=True)
        
        current_step = self.history_horizon_steps - 1

        # Extract data components
        agent_wf_states = data[..., :13]
        agent_bf_states = data[..., 13:26]
        agent_delta_states = data[..., 26:39]
        # agent_wheel_vel_states = data[..., 39:43]  # Unused currently
        agent_actions = data[..., 39:41]
        # Car number is constant across time; use first timestep.
        # Together with dataset_idx it uniquely identifies a car across sources.
        car_number = data[0, 41].long()

        history_horizon = agent_wf_states[:current_step + 1] 
        history_actions = agent_actions[:current_step + 1]
        
        future_horizon = agent_wf_states[current_step:]
        future_actions = agent_actions[current_step:]

        if self.dl_config.is_cmd_vel_in_data:
            agent_cmd_vel = data[..., 39:41]
            history_cmd_vel = agent_cmd_vel[:current_step + 1]
            future_cmd_vel = agent_cmd_vel[current_step:]

        future_horizon_pos_body = agent_bf_states[current_step:, :3]
        future_horizon_quat_rel = agent_bf_states[current_step:, 3:7]
        future_horizon_vel_rel = agent_bf_states[current_step:, 7:10]
        future_horizon_ang_vel_rel = agent_bf_states[current_step:, 10:]

        history_horizon_pos_body = agent_bf_states[:current_step + 1, :3]
        history_horizon_quat_rel = agent_bf_states[:current_step + 1, 3:7]
        history_horizon_vel_rel = agent_bf_states[:current_step + 1, 7:10]
        history_horizon_ang_vel_rel = agent_bf_states[:current_step + 1, 10:]

        current_vel_body = future_horizon_vel_rel[0].unsqueeze(0)
        current_ang_vel_body = future_horizon_ang_vel_rel[0].unsqueeze(0)

        dataset_idx = self.dataset_file_mapping[file_idx]
        
        # Prepare trajectory tensors
        FUT_TRAJ = torch.cat([
            future_horizon_pos_body,
            future_horizon_quat_rel,
            future_horizon_vel_rel,
            future_horizon_ang_vel_rel,
        ], dim=-1)
        
        HIST_TRAJ = torch.cat([
            history_horizon_pos_body, 
            history_horizon_quat_rel, 
            history_horizon_vel_rel, 
            history_horizon_ang_vel_rel
        ], dim=-1)
        
        delta_history = agent_delta_states[:current_step + 1]
        delta_future_full = agent_delta_states[current_step:].clone()
        delta_future_full[0, :3] = 0.0
        delta_future_full[0, 3:7] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            device=data.device,
            dtype=self.dtype,
        )
        delta_future = delta_future_full[..., :13]

        HMAP = self.dummy_hmap_torch        # Final output
        # print(time.time() - start)
        model_input = {
            "history": HIST_TRAJ,
            "future": FUT_TRAJ,
            "future_actions": future_actions,
            "history_actions": history_actions,
            "delta_history": delta_history,
            "delta_future": delta_future,
            "hmap": HMAP,
            'vel_body': current_vel_body,
            'ang_vel_body': current_ang_vel_body,
            "pos_world": agent_wf_states[current_step, :3],
            "quat_wrt_world": agent_wf_states[current_step, 3:7],
            "vel_wrt_world": agent_wf_states[current_step, 7:10],
            "angvel_wrt_world": agent_wf_states[current_step, 10:],
            # "wheel_vels": agent_wheel_vel_states[current_step],
            "dataset_idx": dataset_idx,
            "car_number": car_number,
            # Store full world and body frame data for debugging
            "full_wf_states": agent_wf_states,
            "full_bf_states": agent_bf_states,
            "full_data" : data,
            
        }
        
        if self.dl_config.is_cmd_vel_in_data:
            model_input["history_cmd_vel"] = history_cmd_vel

        if self._dbm_dim:
            # DBM params are constant across time; take first timestep.
            dbm_raw = data[0, self._dbm_slice_start:self._dbm_slice_start + self._dbm_dim]
            model_input["dbm_params_raw"] = dbm_raw
            if self._dbm_mean is not None and self._dbm_std is not None:
                model_input["dbm_params"] = (dbm_raw - self._dbm_mean) / self._dbm_std
            else:
                model_input["dbm_params"] = dbm_raw

        if getattr(self.dl_config, "variable_horizon", False):
            H = self.future_horizon_steps
            sampled_len = int(torch.randint(1, H + 1, (1,)).item())
            future_horizon_mask = torch.zeros(H, dtype=torch.bool)
            future_horizon_mask[:sampled_len] = True
            model_input["future_horizon_mask"] = future_horizon_mask

        return model_input

    def __getstate__(self):
        """Exclude open memmaps when pickling dataset for DataLoader workers."""
        state = self.__dict__.copy()
        if not self.preload_to_ram:
            state["_mmaps"] = [None] * len(self.files)
        return state

    def __setstate__(self, state):
        """Restore dataset state for worker process."""
        self.__dict__.update(state)
        if not self.preload_to_ram:
            if getattr(self, "_mmaps", None) is None or not isinstance(self._mmaps, list):
                self._mmaps = [None] * len(self.files)


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """Split dataset into train/val/test sets."""
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    test_size = int(test_ratio * dataset_size)

    residue = 0
    if train_size + val_size + test_size < dataset_size:
        residue = dataset_size - train_size - val_size - test_size

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size, test_size, residue], generator=generator)
