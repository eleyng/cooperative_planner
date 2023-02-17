""" Some data loading utilities adapted from ctallec's https://github.com/ctallec/world-models/blob/master/data/loaders.py"""
from bisect import bisect
from os import listdir
from os.path import join, isdir
from tqdm import tqdm

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np


class _RolloutDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        transform,
        root,
        seq_len=120,
        H=30,
        skip=5,
        buffer_size=200,
        train=True,
    ):
        self.transform = transform

        self._files = [
            join(root, sd, ssd)
            for sd in listdir(root)
            if isdir(join(root, sd))
            for ssd in listdir(join(root, sd))
        ]

        num_traj = len(self._files)
        n_test = int(np.floor(num_traj * 0.2))
        print("Num traj total: ", num_traj, "Num traj for testing: ", n_test)

        if train:
            self._files = self._files[:-n_test]
        else:
            self._files = self._files[-n_test:]

        self._cum_size = None
        self._buffer = None
        self._buffer_fnames = None
        self._buffer_index = 0
        self._buffer_size = buffer_size
        self.a_low = -1.0
        self.a_hi = 1.0

    def load_next_buffer(self):
        """Loads next buffer, i.e. a subset of the files"""
        if self._buffer_size is None:
            self._buffer_fnames = self._files

        else:
            self._buffer_fnames = self._files[
                self._buffer_index : self._buffer_index + self._buffer_size
            ]
            self._buffer_index += self._buffer_size
            self._buffer_index = self._buffer_index % len(self._files)
        self._buffer = []

        self._cum_size = [0]

        # progress bar
        pbar = tqdm(
            total=len(self._buffer_fnames),
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}",
        )
        pbar.set_description("Loading file buffer ...")

        for f in self._buffer_fnames:

            data = dict(np.load(f, allow_pickle=True))
            self._buffer += [{k: np.copy(v) for k, v in data.items()}]
            self._cum_size += [
                self._cum_size[-1] + self._data_per_sequence(data["actions"].shape[0])
            ]
            pbar.update(1)
        pbar.close()

    def __len__(self):
        if not self._cum_size:
            self.load_next_buffer()
        return self._cum_size[-1]

    def __getitem__(self, i):
        file_index = bisect(self._cum_size, i) - 1
        seq_index = i - self._cum_size[file_index]
        data = self._buffer[file_index].copy()
        return self._get_data(data, seq_index)

    def _get_data(self, data, seq_index):
        pass

    def _data_per_sequence(self, data_length):
        pass


class RolloutSequenceDataset(_RolloutDataset):  # pylint: disable=too-few-public-methods
    """Encapsulates demonstrations.

    Demonstrations should be stored in the datasets/ dir, in the form of npz files,
    each containing a dictionary with the keys:
        - states: (seq_len, state_shape)
        - actions: (seq_len, action_size)
        - table_init: (2,) table initial position in world coordinates
        - table_goal: (2,) table goal position in world coordinates
        - obstacles: (num_obstacles, 2) obstacles positions in world coordinates

    Only a constant number of files (determined by the buffer_size parameter) are loaded
    at a time. Once built, buffers must be loaded with the load_next_buffer method.

    Data are then provided in the form of tuples (states, actions, init_state, table_init, table_goal, obstacles):
    - states: (seq_len // skip, state_shape) states *differences* of the current sequence
    - actions: (seq_len // skip, action_size) actions of the current sequence
    - init_state: (state_shape) initial state of the current sequence
    - table_init: (2,) table initial position in world coordinates
    - table_goal: (2,) table goal position in world coordinates
    - obstacles: (num_obstacles, 2) obstacles positions in world coordinates

    NOTE: seq_len < rollout_len usually

    :args transform: transformation of the observations
    :args root: path to the dataset
    :args seq_len: length of the sequences to return
    :args H: horizon of the sequences to return
    :args skip: number of frames to skip between two consecutive frames
    :args buffer_size: number of files to load at a time
    :args train: if True, uses train data, else test
    """

    def __init__(
        self,
        transform,
        root,
        seq_len=120,
        H=30,
        skip=5,
        buffer_size=200,
        train=True,
    ):
        super().__init__(transform, root, seq_len, H, skip, buffer_size, train)
        self._seq_len = seq_len
        self._H = H
        self.skip = skip

    def _get_data(self, data, seq_index):
        # Load map/dataset info
        table_init = data["table_init"]
        table_goal = data["table_goal"]
        obstacles = data["obstacles"]
        if obstacles.shape[0] < 3:
            obstacles_filler = np.zeros(
                shape=(3 - obstacles.shape[0], 2), dtype=np.float32
            )
            obstacles = np.concatenate(
                (obstacles, obstacles_filler),
                axis=0,
            )

        state_data = data["states"][
            seq_index : seq_index + self._seq_len + 1 : self.skip
        ]

        act_data = data["actions"][
            seq_index : seq_index + self._seq_len + 1 : self.skip
        ]
        act_data = act_data.clip(self.a_low, self.a_hi)
        action = act_data[:-1]
        action = action.astype(np.float32)

        # construct state differences
        init_state = state_data[0, :]
        state_xy = state_data[:, :2]
        state_th = state_data[:, 2:4]
        state_data_ego_pose = np.diff(state_xy, axis=0)
        state_data_ego_th = np.diff(state_th, axis=0)

        # construct goal & obs in ego frame
        goal_lst = np.empty(shape=(state_data_ego_pose.shape[0], 2), dtype=np.float32)
        obs_lst = np.empty(shape=(state_data_ego_pose.shape[0], 2), dtype=np.float32)
        # act_lst = np.empty(shape=(state_data_ego_pose.shape[0], 4), dtype=np.float32)

        for t in range(state_data_ego_pose.shape[0]):
            p_ego2obs_world = state_data[t, 6:] - state_xy[t, :]
            p_ego2goal_world = state_data[t, 4:6] - state_xy[t, :]
            cth = state_data[t, 2]
            sth = state_data[t, 3]

            # rotate goal & obs in ego frame
            obs_lst[t, 0] = cth * p_ego2obs_world[0] + sth * p_ego2obs_world[1]
            obs_lst[t, 1] = -sth * p_ego2obs_world[0] + cth * p_ego2obs_world[1]
            goal_lst[t, 0] = cth * p_ego2goal_world[0] + sth * p_ego2goal_world[1]
            goal_lst[t, 1] = -sth * p_ego2goal_world[0] + cth * p_ego2goal_world[1]

            # actions in ego frame
            # act_lst[t, 0] = cth * action[t, 0] + sth * action[t, 1]
            # act_lst[t, 1] = -sth * action[t, 0] + cth * action[t, 1]
            # act_lst[t, 2] = cth * action[t, 2] + sth * action[t, 3]
            # act_lst[t, 3] = -sth * action[t, 2] + cth * action[t, 3]

        goal_lst = goal_lst / 8
        obs_lst = obs_lst / 8

        state = np.concatenate(
            (
                state_data_ego_pose,
                state_data_ego_th,
                goal_lst,
                obs_lst,
            ),
            axis=1,
        )

        # action = act_lst

        return (
            state,
            action,
            init_state,
            table_init,
            table_goal,
            obstacles,
        )

    def _data_per_sequence(self, data_length):
        return data_length - self._seq_len

    def _check_event(self, action):
        flag = np.all(abs(action) > self._min_action)
        return flag


class RolloutDataModule(pl.LightningDataModule):
    def __init__(
        self,
        transform,
        data_dir,
        seq_len,
        H,
        skip,
        batch_size,
        num_workers,
        test_data,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.skip = skip
        self.H = H
        self.num_workers = num_workers
        self.transform = transform
        self.test_data = test_data
        if self.num_workers > 0:
            self.persistent_workers = True
        else:
            self.persistent_workers = False

        self.train_set = RolloutSequenceDataset(
            self.transform,
            self.data_dir + "/train",
            seq_len=self.seq_len,
            H=self.H,
            skip=self.skip,
            train=True,
            buffer_size=30,
        )
        self.val_set = RolloutSequenceDataset(
            self.transform,
            self.data_dir + "/train",
            seq_len=self.seq_len,
            H=self.H,
            skip=self.skip,
            train=False,
            buffer_size=30,
        )
        self.test_set = RolloutSequenceDataset(
            self.transform,
            self.data_dir + "/test/" + str(self.test_data),
            seq_len=self.seq_len,
            H=self.H,
            skip=self.skip,
            train=True,
            buffer_size=30,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=self.persistent_workers,
        )
