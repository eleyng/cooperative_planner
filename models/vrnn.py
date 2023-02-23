import sys
from os import mkdir
from os.path import isdir, join

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as f

sys.path.append("..")
from utils.learning import frange_cycle_linear
from utils.stats import calc_stats, calculate_fid
from utils.vis import hspace, plot_map, plot_trajectory, vspace

EPS = torch.finfo(torch.float).eps
torch.use_deterministic_algorithms(True, warn_only=True)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class VRNN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters()
        self.name = hparams.name
        if hparams.train:
            self.train_flag = "val"
        else:
            self.train_flag = "test"

        # vrnn model parameters
        self.include_actions = hparams.include_actions
        if hparams.include_actions:
            print("Using actions in inputs.")
        self.skip = hparams.skip
        self.seq_len = hparams.SEQ_LEN // self.skip
        self.H = hparams.H // self.skip
        self.states = hparams.LSIZE
        self.actions = hparams.ASIZE
        self.z_dim = hparams.NLAT
        self.h_dim = hparams.RSIZE
        self.n_layers = hparams.n_layers
        self.batch_size = hparams.BSIZE

        # vrnn training parameters
        self.lr = hparams.lr
        self.lr_min = 1e-6
        self.emb_dim = hparams.emb
        self.weight_decay = hparams.weight_decay
        self.factor = hparams.factor
        self.patience = hparams.patience
        self.epochs = hparams.epochs
        self.q = 1.0
        self.beta_min = 0.0
        self.beta_max = 1.0
        self.cycle = hparams.cycle
        self.R = hparams.R
        self.beta_schedule = frange_cycle_linear(
            self.epochs, self.beta_min, self.beta_max, self.cycle, self.R
        )
        self.beta = self.beta_schedule[0]

        # stats
        self.fid_score_cumsum_sample = 0.0
        self.fid_score_cumsum_dec = 0.0
        self.traj_variance_dec = 0.0
        self.traj_variance_x = 0.0
        self.traj_variance_y = 0.0
        self.traj_variance_theta = 0.0
        self.traj_l2 = 0.0

        # for plotting
        self.fname = None
        self.plot_base_dir = join(hparams.results_dir, hparams.plot_dir)
        if not isdir(hparams.results_dir):
            mkdir(hparams.results_dir)
        if not isdir(self.plot_base_dir):
            mkdir(self.plot_base_dir)

        # define model
        self.xc_dim = self.states
        if self.include_actions:
            self.xc_dim += self.actions
        self.zc_dim = self.z_dim

        self.emb_x = nn.Sequential(
            nn.Linear(self.xc_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )

        self.emb_z = nn.Sequential(
            nn.Linear(self.zc_dim, self.h_dim),
            nn.ReLU(),
        )

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )

        self.enc_mean = nn.Linear(self.h_dim, self.z_dim)

        self.enc_std = nn.Linear(self.h_dim, self.z_dim)

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.xc_dim),
        )

        # recurrent state
        self.gru = nn.GRU(
            self.h_dim + self.h_dim, self.h_dim, self.n_layers, batch_first=True
        )

        # prior
        self.prior = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )
        self.prior_mean = nn.Linear(self.h_dim, self.z_dim)
        self.prior_std = nn.Linear(self.h_dim, self.z_dim)

        if hparams.weight_init == "xavier":
            print("XAVIER UNIFORM")
            self.emb_x.apply(init_weights)
            self.emb_z.apply(init_weights)
            self.enc.apply(init_weights)
            self.enc_mean.apply(init_weights)
            self.enc_std.apply(init_weights)
            self.dec.apply(init_weights)

        elif hparams.weight_init == "default":
            pass
        else:
            pass

    def _reparameterized_sample(self, mu, logvar):
        """
        Reparameterization trick to sample from a Gaussian

        Args:
            mu: mean of the Gaussian torch.tensor of shape (N, T, F)
            logvar: log variance of the Gaussian torch.tensor of shape (N, T, F)

        Returns:
            sample from the Gaussian torch.tensor of shape (N, T, F)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=self.device)
        return eps * std + mu

    def _kld_gauss(self, mean_1, logvar_1, mean_2, logvar_2):
        """Using std to compute KLD

        Args:
            mean_1: mean of the first Gaussian
            logvar_1: log variance of the first Gaussian
            mean_2: mean of the second Gaussian
            logvar_2: log variance of the second Gaussian

        Returns:
            KLD between the two Gaussians
        """
        kld_element = (
            2 * torch.log(torch.sqrt(logvar_2.exp()) + EPS)
            - 2 * torch.log(torch.sqrt(logvar_1.exp()) + EPS)
            + (logvar_1.exp() + (mean_1 - mean_2).pow(2)) / logvar_2.exp()
            - 1
        )
        return 0.5 * torch.sum(kld_element)

    def _l2_loss(self, pred, y):
        """L2 loss"""
        l2 = f.mse_loss(pred, y)
        return l2

    def configure_optimizers(self):
        # Initialize training parameters
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        # optimizer.param_groups[0]["capturable"] = True

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=self.factor,
                    patience=self.patience,
                ),
                "monitor": "val_loss",
            },
        }

    def sample(self, state, action=None, h=None, seq_len=None):
        """Sample from the model

        Args:
            state (torch.tensor): state sequence of shape (N, T, F)
            action (torch.tensor): action sequence of shape (N, T, F)
            h (torch.tensor): hidden state of shape (N, H)
            seq_len (int): sequence length

        Returns:
            sample (torch.tensor): sample from the model of shape (N, T, F)
        """        

        if seq_len is not None:
            T = seq_len // self.skip
        else:
            T = self.seq_len

        samples = []

        # encoder
        if h is None:
            h = torch.zeros(
                self.n_layers, self.batch_size, self.h_dim, device=self.device
            )

        # autoregressive sampling: for the first H // skip steps, we sample from the encoder;
        # then, for the remaining (T-H) // skip steps, we sample from the prior
        for t in range(T):
            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            if t < self.H:
                ins = state[:, t, :]
                if self.include_actions:
                    ins = torch.cat([ins, action[:, t, :]], dim=-1)
                phi_x_t = self.emb_x(ins)

                # encoder
                enc_t = self.enc(torch.cat([phi_x_t, h[-1]], -1))
                enc_mean_t = self.enc_mean(enc_t).squeeze()
                enc_std_t = self.enc_std(enc_t).squeeze()

                # sampling and reparameterization
                z_c = self._reparameterized_sample(enc_mean_t, enc_std_t).view(
                    -1, self.z_dim
                )

                # decoder
                phi_z_t = self.emb_z(z_c)
                dec_t = self.dec(torch.cat([phi_z_t, h[-1]], -1))

            else:
                # sampling and reparameterization
                z_c = self._reparameterized_sample(prior_mean_t, prior_std_t).view(
                    -1, self.z_dim
                )

                # decoder
                phi_z_t = self.emb_z(z_c)
                dec_t = self.dec(torch.cat([phi_z_t, h[-1]], -1))
                phi_x_t = self.emb_x(dec_t)

            # recurrence
            _, h = self.gru(torch.cat([phi_x_t, phi_z_t], -1).unsqueeze(1), h)

            samples.append(dec_t.data)

        samples = torch.stack(samples, dim=1).squeeze()
        if len(samples.shape) == 2:
            samples = samples.unsqueeze(0)
            
        return samples

    def forward(self, state, action=None, h=None):
        all_enc_mean, all_enc_std = [], []
        all_dec_mean = []

        kld_loss = 0
        l2_loss = 0
        loss = 0

        # encoder
        if h is None:
            h = torch.zeros(
                self.n_layers, self.batch_size, self.h_dim, device=self.device
            )

        # autoregressive training: for the first H // skip steps, we sample from the encoder;
        # then, for the remaining (T-H) // skip steps, we sample from the prior
        for t in range(0, self.seq_len):
            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            if t < self.H:
                ins = state[:, t, :]
                if self.include_actions:
                    ins = torch.cat([ins, action[:, t, :]], dim=1)
                phi_x_t = self.emb_x(ins)

                # encoder
                enc_t = self.enc(torch.cat([phi_x_t, h[-1]], -1))
                enc_mean_t = self.enc_mean(enc_t).squeeze()
                enc_std_t = self.enc_std(enc_t).squeeze()

                # sampling and reparameterization
                z_c = self._reparameterized_sample(enc_mean_t, enc_std_t).view(
                    -1, self.z_dim
                )

                # decoder
                phi_z_t = self.emb_z(z_c)
                dec_t = self.dec(torch.cat([phi_z_t, h[-1]], -1))

            else:
                # sampling and reparameterization
                z_c = self._reparameterized_sample(prior_mean_t, prior_std_t).view(
                    -1, self.z_dim
                )

                # decoder
                phi_z_t = self.emb_z(z_c)
                dec_t = self.dec(torch.cat([phi_z_t, h[-1]], -1))
                phi_x_t = self.emb_x(dec_t)

            # recurrence
            _, h = self.gru(torch.cat([phi_x_t, phi_z_t], -1).unsqueeze(1), h)

            # computing losses
            kld_loss_t = self._kld_gauss(
                enc_mean_t, enc_std_t, prior_mean_t, prior_std_t
            )
            kld_loss += kld_loss_t
            if self.include_actions:
                sa = torch.cat([state[:, t, :4], action[:, t, :]], dim=-1)
                dec_t_loss = torch.cat(
                    [dec_t[..., :4], dec_t[..., -self.actions :]], dim=-1
                )
                l2_loss_t = self._l2_loss(dec_t_loss, sa)
                l2_loss += l2_loss_t
            else:
                l2_loss_t = self._l2_loss(dec_t[:, :], state[:, t, :])
                l2_loss += l2_loss_t

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_t)

        recon_loss = l2_loss
        loss = self.beta * kld_loss + recon_loss

        enc_logvar = torch.stack(all_enc_std, dim=1).squeeze()
        enc_mean = torch.stack(all_enc_mean, dim=1).squeeze()
        dec_out = torch.stack(all_dec_mean, dim=1).squeeze()

        return loss, kld_loss, l2_loss, enc_mean, enc_logvar, dec_out, h

    def training_step(self, train_batch, batch_idx):

        (
            state,
            action,
            _,
            _,
            _,
            _,
        ) = train_batch

        if self.include_actions:

            loss, kld_loss, l2_loss, _, _, _, _ = self.forward(state, action)

        else:

            loss, kld_loss, l2_loss, _, _, _, _ = self.forward(state)

        self.log("train_state_pred_l2_state", l2_loss)
        self.log("train_kld", kld_loss)
        self.log("beta", self.beta)
        self.log("train_loss", loss)

        # update beta
        self.beta = self.beta_schedule[self.current_epoch]

        return {"train_loss": loss, "loss": loss}

    def validation_step(self, val_batch, batch_idx):
        (
            state,
            action,
            init_state,
            table_init,
            table_goal,
            obstacles,
        ) = val_batch

        if self.include_actions:
            loss, kld_loss, l2_loss, _, _, _, _ = self.forward(state, action)

        else:
            loss, kld_loss, l2_loss, _, _, _, _ = self.forward(state)

        self.log("val_state_l2", l2_loss)
        self.log("val_kld", kld_loss)
        self.log("beta", self.beta)
        self.log("val_loss", loss)

        if np.random.rand() < 0.9:
            if self.include_actions:
                sample = self.sample(state, action)
            else:
                sample = self.sample(state)

            label = torch.cat([state[:, :, :4], action], dim=-1)
            self.plot(
                label,
                sample,
                init_state=init_state[:, :2],
                table_init=table_init,
                table_goal=table_goal,
                obstacles=obstacles,
                plot_actions=self.include_actions,
            )

        self.log("val_kld", kld_loss)
        self.log("val_loss", loss)

        return {"val_loss": loss, "loss": loss}

    def test_step(self, test_batch, batch_idx):

        (
            state,
            action,
            init_state,
            table_init,
            table_goal,
            obstacles,
        ) = test_batch

        _, _, l2_loss, _, _, _, _ = self.forward(state, action)

        if self.include_actions:
            sample = self.sample(state, action)
        else:
            sample = self.sample(state)

        label = torch.cat([state[:, :, :4], action], dim=-1)
        self.plot(
            label,
            sample,
            init_state=init_state[:, :2],
            table_init=table_init,
            table_goal=table_goal,
            obstacles=obstacles,
            plot_actions=self.include_actions,
        )

        fid_score_sample = calculate_fid(
            state[..., :4].detach().cpu().numpy(),
            sample[..., :4].unsqueeze(0).detach().cpu().numpy(),
        )

        self.fid_score_cumsum_sample += fid_score_sample

        self.traj_variance_theta += calc_stats(
            np.arctan2(
                sample[..., 3].unsqueeze(0).detach().cpu().numpy(),
                sample[..., 2].unsqueeze(0).detach().cpu().numpy(),
            )
        )[1]

        self.traj_variance_x += calc_stats(
            sample[..., 0].unsqueeze(0).detach().cpu().numpy()
        )[1]

        self.traj_variance_y += calc_stats(
            sample[..., 1].unsqueeze(0).detach().cpu().numpy()
        )[1]
        self.traj_l2 += l2_loss

        self.log("test_fid_batch_avg", self.fid_score_cumsum_sample / (batch_idx + 1))
        self.log("test_traj_variance_theta", self.traj_variance_theta / (batch_idx + 1))
        self.log("test_traj_variance_x", self.traj_variance_x / (batch_idx + 1))
        self.log("test_traj_variance_y", self.traj_variance_y / (batch_idx + 1))
        self.log("test_traj_l2", self.traj_l2 / (batch_idx + 1))
        self.log("num_samples", (batch_idx + 1) * self.batch_size)

        return {
            "test_l2_acc": l2_loss,
            "test_fid_sample": fid_score_sample,
            "test_traj_variance_theta": self.traj_variance_theta,
            "test_traj_variance_x": self.traj_variance_x,
            "test_traj_variance_y": self.traj_variance_y,
            "test_traj_l2": self.traj_l2,
            "num_samples": (batch_idx + 1) * self.batch_size,
        }

    def plot(
        self,
        label,
        pred,
        init_state=None,
        table_init=None,
        table_goal=None,
        obstacles=None,
        plot_actions=False,
    ):
        print("Plot during training.")

        init_state = init_state.detach().cpu().numpy()

        fig = plt.figure(figsize=(hspace, vspace), dpi=100)
        table_init_i = table_init
        table_goal_i = table_goal
        obstacles_i = obstacles

        for i in range(1):  # over batch
            init_x = (
                init_state[i, 0]
                + self.q
                * torch.cumsum(label[i, : self.H, 0], dim=0).detach().cpu().numpy()
            )

            init_y = (
                init_state[i, 1]
                + self.q
                * torch.cumsum(label[i, : self.H, 1], dim=0).detach().cpu().numpy()
            )

            table_init_i = table_init[i, :].detach().cpu().numpy()
            table_goal_i = table_goal[i, :].detach().cpu().numpy()
            obstacles_i = obstacles[i, :, :].detach().cpu().numpy()

            plot_map(table_init_i, table_goal_i, obstacles_i, fig=fig, ax=None)

            plt.scatter(init_x, init_y, c="black")
            plot_sample = pred[i, self.H :, :]

            plot_trajectory(
                plot_sample,
                fig,
                init_x,
                init_y,
                self.q,
                c="blue",
                alpha=0.5,
                label="sample",
            )

            plot_trajectory(
                label[i, self.H :, :],
                fig,
                init_x,
                init_y,
                self.q,
                c="green",
                alpha=0.3,
                marker="+",
                label="true",
            )

        file_name = self.name + "_trajvis_{flag}.png".format(flag=self.train_flag)
        plot_name = join(
            self.plot_base_dir,
            self.name,
            file_name,
        )

        if not isdir(join(self.plot_base_dir, self.name)):
            mkdir(join(self.plot_base_dir, self.name))

        plt.savefig(plot_name, dpi=100)
        plt.close()

        if plot_actions:
            fig, axs = plt.subplots(4)
            fig.suptitle("Actions")
            for i in range(1):
                axs[0].plot(label[i, :, -4].detach().cpu().numpy(), label="true")
                axs[0].plot(pred[i, :, -4].detach().cpu().numpy(), label="pred")

                axs[1].plot(label[i, :, -3].detach().cpu().numpy(), label="true")
                axs[1].plot(pred[i, :, -3].detach().cpu().numpy(), label="pred")

                axs[2].plot(label[i, :, -2].detach().cpu().numpy(), label="true")
                axs[2].plot(pred[i, :, -2].detach().cpu().numpy(), label="pred")

                axs[3].plot(label[i, :, -1].detach().cpu().numpy(), label="true")
                axs[3].plot(pred[i, :, -1].detach().cpu().numpy(), label="pred")

            file_name = self.name + "_actions_{flag}.png".format(flag=self.train_flag)
            plot_name = join(
                self.plot_base_dir,
                self.name,
                file_name,
            )
            if not isdir(join(self.plot_base_dir, self.name)):
                mkdir(join(self.plot_base_dir, self.name))

            handles, labels = axs[3].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center")

            plt.savefig(plot_name, dpi=100)
            plt.close()