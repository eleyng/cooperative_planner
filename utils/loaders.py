""" Some data loading utilities """
""" Some data loading utilities """
from bisect import bisect
import random
from os import listdir
from os.path import join, isdir
from re import A
from tqdm import tqdm
import torch
import torch.utils.data
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import numpy as np
import pdb

from utils.misc import WINDOW_W, WINDOW_H


class _RolloutDataset(
    torch.utils.data.Dataset
):  # pylint: disable=too-few-public-methods
    def __init__(
        self,
        transform,
        root,
        seq_len=45,
        H=15,
        skip=1,
        buffer_size=200,
        train=True,
    ):  # pylint: disable=too-many-arguments
        self.transform = transform

        self._files = [
            join(root, sd, ssd)
            for sd in listdir(root)
            if isdir(join(root, sd))
            for ssd in listdir(join(root, sd))
        ]
        # print("files", self._files)

        num_traj = len(self._files)
        n_test = int(np.floor(num_traj * 0.2))
        print("num_traj: ", num_traj, "num_test: ", n_test)

        if train:
            self._files = self._files[:-n_test]
            # print('train', self._files)
        else:
            self._files = self._files[-n_test:]
            # print('test', self._files)

        # self._files = self._files # TEST
        print("num files: ", len(self._files), buffer_size)

        self._cum_size = None
        self._buffer = None
        self._buffer_fnames = None
        self._buffer_index = 0
        self._buffer_size = buffer_size
        self.a_low = -1.0
        self.a_hi = 1.0

    def load_next_buffer(self):
        """Loads next buffer, buffer size < number of rollouts"""
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
            # assert not (data["states"][:, :2] < 2).any(), print(f)
            # print("data", data["states"][:5, :], f)

            # print({k: np.array(v) for k, v in data.items()})

            self._buffer += [{k: np.copy(v) for k, v in data.items()}]
            # print(self._buffer[0]["states"][:2, :])
            self._cum_size += [
                self._cum_size[-1] + self._data_per_sequence(data["actions"].shape[0])
            ]
            pbar.update(1)
        pbar.close()
        print("Cumulative trajectory lengths:", self._cum_size)

    def unstandardize(self, ins, mean, std):
        us = np.multiply(ins, std).add(mean)
        return us

    def standardize(self, ins, mean, std):
        s = np.divide(np.subtract(ins, mean), std)
        return s

    def __len__(self):
        if not self._cum_size:
            self.load_next_buffer()
        return self._cum_size[-1]

    def __getitem__(self, i):
        # binary search through cum_size
        # i = 60
        file_index = bisect(self._cum_size, i) - 1
        seq_index = i - self._cum_size[file_index]
        data = self._buffer[file_index].copy()
        # print("here", self._buffer[0]["states"][:5, :2])
        # print((i, file_index, data["states"]))
        # assert not (data["states"][:, :2] < 2).any(), (i, file_index, data["states"])
        # print("data", len(self._buffer))
        # print("file idx", file_index, seq_index)
        # print("index to buff", self._buffer_fnames[file_index])

        return self._get_data(data, seq_index)

    def _get_data(self, data, seq_index):
        pass

    def _data_per_sequence(self, data_length):
        pass


class RolloutSequenceDataset(_RolloutDataset):  # pylint: disable=too-few-public-methods
    """Encapsulates rollouts.

    Rollouts should be stored in subdirs of the root directory, in the form of npz files,
    each containing a dictionary with the keys:
        - observations: (rollout_len, *obs_shape)
        - actions: (rollout_len, action_size)
        - rewards: (rollout_len,)
        - terminals: (rollout_len,), boolean

     As the dataset is too big to be entirely stored in rams, only chunks of it
     are stored, consisting of a constant number of files (determined by the
     buffer_size parameter).  Once built, buffers must be loaded with the
     load_next_buffer method.

    Data are then provided in the form of tuples (obs, action, reward, terminal, next_obs):
    - obs: (seq_len, *obs_shape)
    - actions: (seq_len, action_size)
    - reward: (seq_len,)
    - terminal: (seq_len,) boolean
    - next_obs: (seq_len, *obs_shape)

    NOTE: seq_len < rollout_len in moste use cases

    :args root: root directory of data sequences
    :args seq_len: number of timesteps extracted from each rollout
    :args transform: transformation of the observations
    :args train: if True, train data, else test
    """

    def __init__(
        self,
        transform,
        root,
        seq_len=45,
        H=15,
        skip=1,
        buffer_size=200,
        train=True,
    ):  # pylint: disable=too-many-arguments
        super().__init__(
            transform, root, seq_len, H, skip, buffer_size, train
        )
        self._seq_len = seq_len
        self._H = H
        self.skip = skip

    def _get_data(self, data, seq_index):
        # assert not (data["states"][:, :2] < 2).any(), (seq_index, data["states"][:, :2])

        # y_var = data["conditioning_var"][seq_index : seq_index + self._seq_len]
        # print("data", data["states"][:, 0])

        # Load map/dataset info
        # map_file_path = data["file_identifier"]
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
            seq_index : seq_index
            + self._seq_len
            + 1 : self.skip
            # np.array([0, 1, 8, 9]),  # , 3, 4, 5, 6, 7]),
        ]

        states_wf = state_data[:-1, :4]

        act_data = data["actions"][
            seq_index : seq_index + self._seq_len + 1 : self.skip
        ]

        action, next_action = act_data[:-1], act_data[1:]

        # clip
        act_data = act_data.clip(self.a_low, self.a_hi)

        # convert to float
        action = action.astype(np.float32)
        next_action = next_action.astype(np.float32)
        # print("state data", state_data.shape)
        # print("before tf state_data", state_data[:, :4])

        # assert len(state_data) == self._seq_len + 1, len(state_data)

        # process states to use relative x, y offset

        # Uncomment for state differences
        q = 1

        init_state = state_data[0, :]
        state_xy = state_data[:, :2]
        state_th = state_data[:, 2:4]
        state_data_ego_pose = np.diff(state_xy, axis=0) / q
        state_data_ego_th = np.diff(state_th, axis=0)

        # print("state_th", state_th)

        goal_lst = np.empty(shape=(state_data_ego_pose.shape[0], 2), dtype=np.float32)
        obs_lst = np.empty(shape=(state_data_ego_pose.shape[0], 2), dtype=np.float32)
        act_lst = np.empty(shape=(state_data_ego_pose.shape[0], 4), dtype=np.float32)
        # pose_diff_lst = np.empty(
        #     shape=(state_data_ego_pose.shape[0], 2), dtype=np.float32
        # )
        # th_diff_lst = np.empty(shape=(state_data_ego_th.shape[0], 2), dtype=np.float32)
        for t in range(state_data_ego_pose.shape[0]):
            p_ego2obs_world = state_data[t, 6:] - state_xy[t, :]
            # print("p_ego2obs_world", p_ego2obs_world.shape)
            p_ego2goal_world = state_data[t, 4:6] - state_xy[t, :]
            # print("p_ego2goal_world", p_ego2goal_world)
            # print("p_ego2obs_world", p_ego2obs_world)

            cth = state_data[t, 2]
            sth = state_data[t, 3]
            # goal & obs in ego frame
            obs_lst[t, 0] = cth * p_ego2obs_world[0] + sth * p_ego2obs_world[1]
            obs_lst[t, 1] = -sth * p_ego2obs_world[0] + cth * p_ego2obs_world[1]

            goal_lst[t, 0] = cth * p_ego2goal_world[0] + sth * p_ego2goal_world[1]
            goal_lst[t, 1] = -sth * p_ego2goal_world[0] + cth * p_ego2goal_world[1]

            # actions in ego frame
            act_lst[t, 0] = cth * action[t, 0] + sth * action[t, 1]
            act_lst[t, 1] = -sth * action[t, 0] + cth * action[t, 1]
            act_lst[t, 2] = cth * action[t, 2] + sth * action[t, 3]
            act_lst[t, 3] = -sth * action[t, 2] + cth * action[t, 3]
            # pose in ego frame
            # print(
            #     t,
            #     "state_data_ego_pose",
            #     state_data_ego_pose[t, 0].shape,
            #     cth,
            #     pose_diff_lst.shape,
            #     pose_diff_lst[t, 0].shape,
            #     cth * state_data_ego_pose[t, 0] + sth * state_data_ego_pose[t, 1],
            # )
            # pose_diff_lst[t, 0] = (
            #     cth * state_data_ego_pose[t, 0] + sth * state_data_ego_pose[t, 1]
            # )
            # pose_diff_lst[t, 1] = (
            #     -sth * state_data_ego_pose[t, 0] + cth * state_data_ego_pose[t, 1]
            # )

            # # th in ego frame
            # th_diff_lst[t, 0] = (
            #     cth * state_data_ego_th[t, 0] + sth * state_data_ego_th[t, 1]
            # )
            # th_diff_lst[t, 1] = (
            #     -sth * state_data_ego_th[t, 0] + cth * state_data_ego_th[t, 1]
            # )

        goal_lst = goal_lst / 8
        obs_lst = obs_lst / 8

        # print("goal_lst", goal_lst)
        # print("obs_lst", obs_lst)
        # print("state_data", state_xy)

        # state_data = np.concatenate(
        #     (state_data_ego_pose,),
        #     axis=1,
        # )

        # state = state_data_ego_pose
        state = np.concatenate(
            (
                # pose_diff_lst,  #
                # th_diff_lst,
                state_data_ego_pose,
                state_data_ego_th,
                # states_wf,
                goal_lst,
                obs_lst,
            ),
            axis=1,
        )

        action = act_lst
        # print("after tf state_datea", state.shape)

        ## Can safely ignore belwo for now
        next_state = state_data  ## WRONG, fix this
        y_var = state_data

        # y0 = y_var[:, 0] / WINDOW_W
        # y1 = y_var[:, 1] / WINDOW_H
        # y2 = y_var[:, 2] / (3 / 2 * np.pi)
        # y3 = y_var[:, 3] / WINDOW_W
        # y4 = y_var[:, 4] / WINDOW_H

        # print("state start", data["states"])

        # act_data = data["actions"][seq_index : seq_index + self._seq_len]
        # action, next_action = act_data[: self._H], act_data[self._H :]

        # reward = data["rewards"][seq_index + 1 : seq_index + self._seq_len + 1].astype(
        #     np.float32
        # )

        # normalize
        # assert not (state_data[:, :2] < 2).any(), state_data ## ABSOLUTE STATES

        # if loading data for finding train stats
        if self.transform == "min-max":

            sdx = state[:, 0] / WINDOW_W
            sdy = state[:, 1] / WINDOW_H
            sdct = (state[:, 2] - (-1)) / (1 - (-1))
            sdst = (state[:, 3] - (-1)) / (1 - (-1))
            d2g = state[:, 4] / np.linalg.norm(
                np.array([WINDOW_W, WINDOW_H]) - np.array([0, 0])
            )
            d2o = state[:, 5:] / np.linalg.norm(
                np.array([WINDOW_W, WINDOW_H]) - np.array([0, 0])
            )

            sdx = np.expand_dims(sdx, axis=1)
            sdy = np.expand_dims(sdy, axis=1)
            sdct = np.expand_dims(sdct, axis=1)
            sdst = np.expand_dims(sdst, axis=1)
            d2g = np.expand_dims(d2g, axis=1)

            state = np.concatenate((sdx, sdy, sdct, sdst, d2g, d2o), axis=1)

            y0 = y_var[:, 0] / WINDOW_W
            y1 = y_var[:, 1] / WINDOW_H
            y2 = y_var[:, 2] / (3 / 2 * np.pi)
            y3 = y_var[:, 3] / WINDOW_W
            y4 = y_var[:, 4] / WINDOW_H

            y0 = np.expand_dims(y0, axis=1)
            y1 = np.expand_dims(y1, axis=1)
            y2 = np.expand_dims(y2, axis=1)
            y3 = np.expand_dims(y3, axis=1)
            y4 = np.expand_dims(y4, axis=1)

            y_var = np.concatenate((y0, y1, y2, y3, y4, y_var[:, 5:]), axis=1)

            return (
                state,
                action,
                next_state,
                next_action,
                y_var,
            )
        else:
            assert state.shape[0] == action.shape[0]
            return (
                state,
                action,
                next_state,
                next_action,
                y_var,
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
        print("NUMWORKERS", num_workers, "TRANSFORM", transform)
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
        # self.val_set = self.train_set
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
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=self.persistent_workers,
        )