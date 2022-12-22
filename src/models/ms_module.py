from typing import Any, List

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
import numpy as np

from 
from src.utils.metrics import TranslationalError, RotationalError
from src.utils.MoCap_Solver.utils import (
    vertex_loss,
    angle_loss,
    HuberLoss,
    Criterion_EE,
)


def fk_ts(X_t, topology):
    X_t_glob = X_t.clone()
    for i in range(1, len(topology)):
        X_t_glob[:, i, :] = X_t_glob[:, i, :] + X_t[:, topology[i], :]
    return X_t_glob


class MSNoEncLitModule(LightningModule):
    """Example of LightningModule for template skeleton autoencoder.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        encoder_net: torch.nn.Module,
        decoder_net: torch.nn.Module,
        ts_decoder_net: torch.nn.Module,
        mc_decoder_net: torch.nn.Module,
        mo_decoder_net: torch.nn.Module,
        fk,
        skin,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        topology: list,
        data_dir: str = "data/",
        beta_1: float = 20.0,
        beta_2: float = 50.0,
        beta_3: float = 1000.0,
        beta_4: float = 1.0,
        beta_5: float = 2.0,
        beta_6: float = 10.0,
        beta_7: float = 5000.0,
        beta_8: float = 1.0,
        beta_9: float = 100.0,
        beta_10: float = 100.0,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "encoder_net", "decoder_net", "ts_decoder_net", "mc_decoder_net", "mo_decoder_net"
            ],
        )

        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.ts_decoder_net = ts_decoder_net
        self.mc_decoder_net = mc_decoder_net
        self.mo_decoder_net = mo_decoder_net
        self.fk = fk
        self.skin = skin

        clean_markers_stat_dir = (
            data_dir + "/MS_Synthetic_preprocessed/clean_markers_statistics.npy"
        )

        # Hard coded !!!!!
        first_rot_stat_dir = (
            data_dir + "/ours_Synthetic/msalign_first_rot_statistics.npy"
        )
        ts_stat_dir = data_dir + "/MS_Synthetic_preprocessed/ts_statistics.npy"
        mc_stat_dir = data_dir + "/MS_Synthetic_preprocessed/mc_statistics.npy"

        # Hard coded !!!!!
        mo_stat_dir = data_dir + "/ours_Synthetic/msalign_mo_statistics.npy"
        skinning_weights_dir = (
            data_dir + "/MS_Synthetic_preprocessed/skinning_weights.npy"
        )

        clean_markers_mean, clean_markers_std = torch.tensor(
            np.load(clean_markers_stat_dir), dtype=torch.float32
        )
        first_rot_mean, first_rot_std = torch.tensor(
            np.load(first_rot_stat_dir), dtype=torch.float32
        )
        ts_data_mean, ts_data_std = torch.tensor(
            np.load(ts_stat_dir), dtype=torch.float32
        )
        mc_data_mean, mc_data_std = torch.tensor(
            np.load(mc_stat_dir), dtype=torch.float32
        )
        mo_data_mean, mo_data_std = torch.tensor(
            np.load(mo_stat_dir), dtype=torch.float32
        )[:, None, :, None]
        skinning_weights = torch.tensor(
            np.load(skinning_weights_dir), dtype=torch.float32
        )

        lambda_para = torch.tensor(
            self.encoder_net.lambda_para[None, None, ...], dtype=torch.float32
        )
        lambda_para_jt = torch.tensor(
            self.encoder_net.lambda_jt_para[None, None, ...], dtype=torch.float32
        )
        lambda_para_jt1 = torch.tensor(
            self.encoder_net.lambda_jt_para[None, None, ..., None], dtype=torch.float32
        )

        self.register_buffer("clean_markers_mean", clean_markers_mean)
        self.register_buffer("clean_markers_std", clean_markers_std)
        self.register_buffer("first_rot_mean", first_rot_mean)
        self.register_buffer("first_rot_std", first_rot_std)
        self.register_buffer("ts_data_mean", ts_data_mean)
        self.register_buffer("ts_data_std", ts_data_std)
        self.register_buffer("mc_data_mean", mc_data_mean)
        self.register_buffer("mc_data_std", mc_data_std)
        self.register_buffer("mo_data_mean", mo_data_mean)
        self.register_buffer("mo_data_std", mo_data_std)
        self.register_buffer("skinning_weights", skinning_weights)

        self.register_buffer("lambda_para", lambda_para)
        self.register_buffer("lambda_para_jt", lambda_para_jt)
        self.register_buffer("lambda_para_jt1", lambda_para_jt1)

        # loss function
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_huber = HuberLoss(100)
        self.criterion_l1 = torch.nn.L1Loss()
        self.criterion_ee = Criterion_EE(1, torch.nn.MSELoss())
        self.criterion_ang = angle_loss()
        self.criterion_ver = vertex_loss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_jpe = TranslationalError(in_metric="m", out_metric="mm")
        self.train_joe = RotationalError()
        self.train_mpe = TranslationalError(in_metric="m", out_metric="mm")
        self.val_jpe = TranslationalError(in_metric="m", out_metric="mm")
        self.val_joe = RotationalError()
        self.val_mpe = TranslationalError(in_metric="m", out_metric="mm")
        self.val_err = MeanMetric()
        self.test_jpe = TranslationalError(in_metric="m", out_metric="mm")
        self.test_joe = RotationalError()
        self.test_mpe = TranslationalError(in_metric="m", out_metric="mm")

        # for logging best so far validation accuracy
        self.val_err_best = MinMetric()

    def forward(self, x: torch.Tensor):
        output = self.encoder_net(x)
        return output

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_err_best.reset()

    def step(self, batch: Any):
        (
            raw_marker,
            clean_marker,
            skeleton_pos,
            motion,
            offsets,
            marker_config,
            first_rot,
            of_code,
            mc_code,
            transform,
        ) = batch

        latent = self.encoder_net(raw_marker)
        l_c, l_m, l_t = self.decoder_net(latent)
        # Check this part?
        l_m0 = l_m[:, 256:]
        l_m0 = l_m0.view(-1, 112, 16)
        l_m1_norm = l_m[:, :256].view(-1, 64, 4)
        out_first_rot = l_m1_norm * self.first_rot_std + self.first_rot_mean
        out_first_rot = F.normalize(out_first_rot, dim=2)

        out_offset = self.ts_decoder_net(l_t)[::-1]
        # Hard coded
        out_offset[0] = out_offset[0].view(-1, 24, 3)
        out_offset[0] = out_offset[0] * self.ts_data_std + self.ts_data_mean
        res_offset_output_input = (out_offset[0] - self.ts_data_mean) / self.ts_data_std

        out_markerconf = self.mc_decoder_net(l_c, res_offset_output_input, l_t)
        res_markerconf_all = (out_markerconf * self.mc_data_std) + self.mc_data_mean
        res_markerconf = res_markerconf_all[:, :, :, :3]

        motion_code = l_m0.view(-1, 112, 16)
        out_motion = self.mo_decoder_net(motion_code, out_offset)
        res_motion = (out_motion * self.mo_data_std) + self.mo_data_mean

        res_pos, res_rot = self.fk.forward_from_raw2(
            res_motion, out_offset[0], out_first_rot, world=True, quater=True
        )
        res_markers = self.skin.skinning(
            res_markerconf, self.skinning_weights[None, ...], res_rot, res_pos
        )

        res_first_rot_xform = self.fk.transform_from_quaternion(out_first_rot)
        first_rot_xform = self.fk.transform_from_quaternion(first_rot)

        mrk_pos_loss = self.criterion_huber(
            clean_marker * self.lambda_para, res_markers * self.lambda_para
        )
        skel_pos_loss = self.criterion_huber(
            skeleton_pos * self.lambda_para_jt, res_pos * self.lambda_para_jt
        )
        first_rot_xform_loss = self.criterion_huber(
            first_rot_xform, res_first_rot_xform
        )
        motion_xform_loss = self.criterion_huber(
            transform * self.lambda_para_jt1, res_rot * self.lambda_para_jt1
        )
        motion_loss = self.criterion_huber(res_motion[..., :-3], motion[..., :-3])
        motion_trans_loss = self.criterion_huber(
            res_motion[:, -3:, :].permute(0, 2, 1), motion[:, -3:, :].permute(0, 2, 1)
        )
        marker_config_loss = self.criterion_huber(
            marker_config[:, :, :, :3], res_markerconf
        )
        offset_loss = self.criterion_huber(
            offsets * self.lambda_para_jt, out_offset[0] * self.lambda_para_jt
        )

        loss = (
            mrk_pos_loss * self.hparams.beta_1
            + skel_pos_loss * self.hparams.beta_2
            + first_rot_xform_loss * self.hparams.beta_3
            + motion_xform_loss * self.hparams.beta_4
            # + offset_latent_loss * self.hparams.beta_5
            + motion_loss * self.hparams.beta_6
            + motion_trans_loss * self.hparams.beta_7
            # + marker_latent_loss * self.hparams.beta_8
            + marker_config_loss * self.hparams.beta_9
            + offset_loss * self.hparams.beta_10
        )

        return (
            loss,
            clean_marker,
            res_markers,
            transform,
            res_rot,
            skeleton_pos,
            res_pos,
        )

    def training_step(self, batch: Any, batch_idx: int):
        (
            loss,
            clean_markers,
            res_markers,
            transform,
            res_rot,
            skeleton_pos,
            res_pos,
        ) = self.step(batch)

        # log train metrics
        joint_pos_error = self.train_jpe(skeleton_pos, res_pos)
        joint_ori_error = self.train_joe(transform, res_rot)
        marker_pos_error = self.train_mpe(clean_markers, res_markers)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "train/jpe", joint_pos_error, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "train/joe", joint_ori_error, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "train/mpe", marker_pos_error, on_step=False, on_epoch=True, prog_bar=False
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_jpe.reset()
        self.train_joe.reset()
        self.train_mpe.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        (
            loss,
            clean_markers,
            res_markers,
            transform,
            res_rot,
            skeleton_pos,
            res_pos,
        ) = self.step(batch)
        batch_size = clean_markers.shape[0]

        # log val metrics
        joint_pos_error = self.val_jpe(skeleton_pos, res_pos)
        joint_ori_error = self.val_joe(transform, res_rot)
        marker_pos_error = self.val_mpe(clean_markers, res_markers)
        self.val_err.update(loss, weight=batch_size)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "val/jpe", joint_pos_error, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val/joe", joint_ori_error, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val/mpe", marker_pos_error, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        err = self.val_err.compute()  # get val accuracy from current epoch
        self.val_err_best.update(err)
        self.log(
            "val/err_best", self.val_err_best.compute(), on_epoch=True, prog_bar=True
        )
        self.val_err.reset()
        self.val_jpe.reset()
        self.val_joe.reset()
        self.val_mpe.reset()

    def test_step(self, batch: Any, batch_idx: int):
        (
            loss,
            clean_markers,
            res_markers,
            transform,
            res_rot,
            skeleton_pos,
            res_pos,
        ) = self.step(batch)

        # log test metrics
        joint_pos_error = self.test_jpe(skeleton_pos, res_pos)
        joint_ori_error = self.test_joe(transform, res_rot)
        marker_pos_error = self.test_mpe(clean_markers, res_markers)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/jpe", joint_pos_error, on_step=False, on_epoch=True)
        self.log("test/joe", joint_ori_error, on_step=False, on_epoch=True)
        self.log("test/mpe", marker_pos_error, on_step=False, on_epoch=True)

        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        self.test_jpe.reset()
        self.test_joe.reset()
        self.test_mpe.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return [optimizer], [scheduler]


class MSNoEncAlignerLitModule(LightningModule):
    """Example of LightningModule for template skeleton autoencoder.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        aligner_net: torch.nn.Module,
        encoder_net: torch.nn.Module,
        decoder_net: torch.nn.Module,
        ts_decoder_net: torch.nn.Module,
        mc_decoder_net: torch.nn.Module,
        mo_decoder_net: torch.nn.Module,
        fk,
        skin,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        topology: list,
        data_dir: str = "data/",
        beta_1: float = 20.0,
        beta_2: float = 50.0,
        beta_3: float = 1000.0,
        beta_4: float = 1.0,
        beta_5: float = 2.0,
        beta_6: float = 10.0,
        beta_7: float = 5000.0,
        beta_8: float = 1.0,
        beta_9: float = 100.0,
        beta_10: float = 100.0,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "aligner_net", "encoder_net", "decoder_net", "ts_decoder_net", "mc_decoder_net", "mo_decoder_net"
            ],
        )

        self.aligner_net = aligner_net
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.ts_decoder_net = ts_decoder_net
        self.mc_decoder_net = mc_decoder_net
        self.mo_decoder_net = mo_decoder_net
        self.fk = fk
        self.skin = skin

        clean_markers_stat_dir = (
            data_dir + "/MS_Synthetic_preprocessed/clean_markers_statistics.npy"
        )

        ts_stat_dir = data_dir + "/MS_Synthetic_preprocessed/ts_statistics.npy"
        mc_stat_dir = data_dir + "/MS_Synthetic_preprocessed/mc_statistics.npy"

        # Hard coded !!!!!
        mo_stat_dir = data_dir + "/ours_Synthetic/msalign_mo_statistics.npy"
        skinning_weights_dir = (
            data_dir + "/MS_Synthetic_preprocessed/skinning_weights.npy"
        )

        clean_markers_mean, clean_markers_std = torch.tensor(
            np.load(clean_markers_stat_dir), dtype=torch.float32
        )
        first_rot_mean, first_rot_std = torch.tensor(
            np.load(first_rot_stat_dir), dtype=torch.float32
        )
        ts_data_mean, ts_data_std = torch.tensor(
            np.load(ts_stat_dir), dtype=torch.float32
        )
        mc_data_mean, mc_data_std = torch.tensor(
            np.load(mc_stat_dir), dtype=torch.float32
        )
        mo_data_mean, mo_data_std = torch.tensor(
            np.load(mo_stat_dir), dtype=torch.float32
        )[:, None, :, None]
        mo_data_mean[..., -3] = 0.0
        mo_data_std[..., -3] = 1.0
        skinning_weights = torch.tensor(
            np.load(skinning_weights_dir), dtype=torch.float32
        )

        lambda_para = torch.tensor(
            self.encoder_net.lambda_para[None, None, ...], dtype=torch.float32
        )
        lambda_para_jt = torch.tensor(
            self.encoder_net.lambda_jt_para[None, None, ...], dtype=torch.float32
        )
        lambda_para_jt1 = torch.tensor(
            self.encoder_net.lambda_jt_para[None, None, ..., None], dtype=torch.float32
        )

        self.register_buffer("clean_markers_mean", clean_markers_mean)
        self.register_buffer("clean_markers_std", clean_markers_std)
        self.register_buffer("ts_data_mean", ts_data_mean)
        self.register_buffer("ts_data_std", ts_data_std)
        self.register_buffer("mc_data_mean", mc_data_mean)
        self.register_buffer("mc_data_std", mc_data_std)
        self.register_buffer("mo_data_mean", mo_data_mean)
        self.register_buffer("mo_data_std", mo_data_std)
        self.register_buffer("skinning_weights", skinning_weights)

        self.register_buffer("lambda_para", lambda_para)
        self.register_buffer("lambda_para_jt", lambda_para_jt)
        self.register_buffer("lambda_para_jt1", lambda_para_jt1)

        # loss function
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_huber = HuberLoss(100)
        self.criterion_l1 = torch.nn.L1Loss()
        self.criterion_ee = Criterion_EE(1, torch.nn.MSELoss())
        self.criterion_ang = angle_loss()
        self.criterion_ver = vertex_loss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_jpe = TranslationalError(in_metric="m", out_metric="mm")
        self.train_joe = RotationalError()
        self.train_mpe = TranslationalError(in_metric="m", out_metric="mm")
        self.val_jpe = TranslationalError(in_metric="m", out_metric="mm")
        self.val_joe = RotationalError()
        self.val_mpe = TranslationalError(in_metric="m", out_metric="mm")
        self.val_err = MeanMetric()
        self.test_jpe = TranslationalError(in_metric="m", out_metric="mm")
        self.test_joe = RotationalError()
        self.test_mpe = TranslationalError(in_metric="m", out_metric="mm")

        # for logging best so far validation accuracy
        self.val_err_best = MinMetric()

    def forward(self, x: torch.Tensor):
        output = self.encoder_net(x)
        return output

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_err_best.reset()

    def step(self, batch: Any):
        (
            raw_marker,
            clean_marker,
            skeleton_pos,
            motion,
            offsets,
            marker_config,
            first_rot,
            of_code,
            mc_code,
            transform,
        ) = batch

        latent = self.encoder_net(raw_marker)
        l_c, l_m, l_t = self.decoder_net(latent)
        # Check this part?
        l_m0 = l_m[:, 256:]
        l_m0 = l_m0.view(-1, 112, 16)
        l_m1_norm = l_m[:, :256].view(-1, 64, 4)
        out_first_rot = F.normalize(l_m1_norm, dim=2)

        out_offset = self.ts_decoder_net(l_t)[::-1]
        # Hard coded
        out_offset[0] = out_offset[0].view(-1, 24, 3)
        out_offset[0] = out_offset[0] * self.ts_data_std + self.ts_data_mean
        res_offset_output_input = (out_offset[0] - self.ts_data_mean) / self.ts_data_std

        out_markerconf = self.mc_decoder_net(l_c, res_offset_output_input, l_t)
        res_markerconf_all = (out_markerconf * self.mc_data_std) + self.mc_data_mean
        res_markerconf = res_markerconf_all[:, :, :, :3]

        motion_code = l_m0.view(-1, 112, 16)
        out_motion = self.mo_decoder_net(motion_code, out_offset)
        res_motion = (out_motion * self.mo_data_std) + self.mo_data_mean

        res_pos, res_rot = self.fk.forward_from_raw2(
            res_motion, out_offset[0], out_first_rot, world=True, quater=True
        )
        res_markers = self.skin.skinning(
            res_markerconf, self.skinning_weights[None, ...], res_rot, res_pos
        )

        res_first_rot_xform = self.fk.transform_from_quaternion(out_first_rot)
        first_rot_xform = self.fk.transform_from_quaternion(first_rot)

        mrk_pos_loss = self.criterion_huber(
            clean_marker * self.lambda_para, res_markers * self.lambda_para
        )
        skel_pos_loss = self.criterion_huber(
            skeleton_pos * self.lambda_para_jt, res_pos * self.lambda_para_jt
        )
        first_rot_xform_loss = self.criterion_huber(
            first_rot_xform, res_first_rot_xform
        )
        motion_xform_loss = self.criterion_huber(
            transform * self.lambda_para_jt1, res_rot * self.lambda_para_jt1
        )
        motion_loss = self.criterion_huber(res_motion[..., :-3], motion[..., :-3])
        motion_trans_loss = self.criterion_huber(
            res_motion[:, -3:, :].permute(0, 2, 1), motion[:, -3:, :].permute(0, 2, 1)
        )
        marker_config_loss = self.criterion_huber(
            marker_config[:, :, :, :3], res_markerconf
        )
        offset_loss = self.criterion_huber(
            offsets * self.lambda_para_jt, out_offset[0] * self.lambda_para_jt
        )

        loss = (
            mrk_pos_loss * self.hparams.beta_1
            + skel_pos_loss * self.hparams.beta_2
            + first_rot_xform_loss * self.hparams.beta_3
            + motion_xform_loss * self.hparams.beta_4
            # + offset_latent_loss * self.hparams.beta_5
            + motion_loss * self.hparams.beta_6
            + motion_trans_loss * self.hparams.beta_7
            # + marker_latent_loss * self.hparams.beta_8
            + marker_config_loss * self.hparams.beta_9
            + offset_loss * self.hparams.beta_10
        )

        return (
            loss,
            clean_marker,
            res_markers,
            transform,
            res_rot,
            skeleton_pos,
            res_pos,
        )

    def training_step(self, batch: Any, batch_idx: int):
        (
            loss,
            clean_markers,
            res_markers,
            transform,
            res_rot,
            skeleton_pos,
            res_pos,
        ) = self.step(batch)

        # log train metrics
        joint_pos_error = self.train_jpe(skeleton_pos, res_pos)
        joint_ori_error = self.train_joe(transform, res_rot)
        marker_pos_error = self.train_mpe(clean_markers, res_markers)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "train/jpe", joint_pos_error, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "train/joe", joint_ori_error, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "train/mpe", marker_pos_error, on_step=False, on_epoch=True, prog_bar=False
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_jpe.reset()
        self.train_joe.reset()
        self.train_mpe.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        (
            loss,
            clean_markers,
            res_markers,
            transform,
            res_rot,
            skeleton_pos,
            res_pos,
        ) = self.step(batch)
        batch_size = clean_markers.shape[0]

        # log val metrics
        joint_pos_error = self.val_jpe(skeleton_pos, res_pos)
        joint_ori_error = self.val_joe(transform, res_rot)
        marker_pos_error = self.val_mpe(clean_markers, res_markers)
        self.val_err.update(loss, weight=batch_size)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "val/jpe", joint_pos_error, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val/joe", joint_ori_error, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val/mpe", marker_pos_error, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        err = self.val_err.compute()  # get val accuracy from current epoch
        self.val_err_best.update(err)
        self.log(
            "val/err_best", self.val_err_best.compute(), on_epoch=True, prog_bar=True
        )
        self.val_err.reset()
        self.val_jpe.reset()
        self.val_joe.reset()
        self.val_mpe.reset()

    def test_step(self, batch: Any, batch_idx: int):
        (
            loss,
            clean_markers,
            res_markers,
            transform,
            res_rot,
            skeleton_pos,
            res_pos,
        ) = self.step(batch)

        # log test metrics
        joint_pos_error = self.test_jpe(skeleton_pos, res_pos)
        joint_ori_error = self.test_joe(transform, res_rot)
        marker_pos_error = self.test_mpe(clean_markers, res_markers)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/jpe", joint_pos_error, on_step=False, on_epoch=True)
        self.log("test/joe", joint_ori_error, on_step=False, on_epoch=True)
        self.log("test/mpe", marker_pos_error, on_step=False, on_epoch=True)

        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        self.test_jpe.reset()
        self.test_joe.reset()
        self.test_mpe.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "ms_ts.yaml")
    _ = hydra.utils.instantiate(cfg)
