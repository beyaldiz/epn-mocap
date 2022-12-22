from typing import Any, List

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import MinMetric, MeanMetric

import vgtk.so3conv.functional as L

from src.models.modules.epn_mocap_net import MocapNet
from src.utils.metrics import TranslationalError, RotationalError
from src.utils.transform import symmetric_orthogonalization

class HuberLoss(nn.Module):
    def __init__(self, delta):
        super(HuberLoss, self).__init__()
        self.HUBER_DELTA = delta

    def forward(self, input, target):
        error_mat = input - target
        _error_ = torch.sqrt(torch.sum(error_mat **2))
        HUBER_DELTA = self.HUBER_DELTA
        switch_l = _error_<HUBER_DELTA
        switch_2 = _error_>=HUBER_DELTA
        x = switch_l * (0.5* _error_**2 ) + switch_2 * (0.5* HUBER_DELTA**2 + HUBER_DELTA*(_error_-HUBER_DELTA))
        return x


class EPNMocapLitModel(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        mlps=[[32,32], [64,64], [128,128], [256, 256]],
        out_mlps=[128, 128],
        strides=[2, 2, 2, 1],
        initial_radius_ratio = 0.2,
        sampling_ratio = 0.8,
        sampling_density = 0.5,
        kernel_density = 1,
        kernel_multiplier = 2,
        sigma_ratio= 0.5, # 1e-3, 0.68
        xyz_pooling = None, # None, 'no-stride',
        lr = 0.0005,
        lr_decay = 0.9995
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = MocapNet(
            mlps=mlps,
            out_mlps=out_mlps,
            strides=strides,
            initial_radius_ratio=initial_radius_ratio,
            sampling_ratio=sampling_ratio,
            sampling_density=sampling_density,
            kernel_density=kernel_density,
            kernel_multiplier=kernel_multiplier,
            sigma_ratio=sigma_ratio,
            xyz_pooling=xyz_pooling,
        )
        
        self.register_buffer("anchor", torch.tensor(L.get_anchors(), dtype=torch.float32))

        # loss function
        self.criterion_huber = HuberLoss(100)
        self.criterion_cls = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_jpe = TranslationalError(in_metric="m", out_metric="mm")
        self.train_joe = RotationalError()
        self.val_jpe = TranslationalError(in_metric="m", out_metric="mm")
        self.val_joe = RotationalError()
        self.val_err = MeanMetric()
        self.test_jpe = TranslationalError(in_metric="m", out_metric="mm")
        self.test_joe = RotationalError()

        # for logging best so far validation accuracy
        self.val_err_best = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_err_best.reset()

    def step(self, batch: Any):
        (
            raw_markers,
            J_R_gt,
            J_t_gt,
            R_label_gt,
            R_rel_gt
        ) = batch['xyz'], batch['J_R'], batch['J_t'], batch['R_label'], batch['R_rel']

        output = self.model(raw_markers.reshape(-1, 64*56, 3))
        R_label_pred = output['1'].argmax(-1)
        out_R = output['R'].reshape(-1, 64, 24, 3, 3, 60).permute(0, 1, 2, 5, 3, 4).contiguous()
        out_R = symmetric_orthogonalization(out_R)
        out_R = out_R.reshape(-1, 60, 3, 3)
        out_R_glob = self.anchor @ out_R

        J_R_exp_rel = out_R[torch.arange(out_R.shape[0]), R_label_gt.reshape(-1)].reshape(-1, 64, 24, 3, 3)
        J_R = out_R_glob[torch.arange(out_R_glob.shape[0]), R_label_pred.reshape(-1)].reshape(-1, 64, 24, 3, 3)
        J_t = output['T'].mean(-1).reshape(-1, 64, 24, 3)
        R_label_scores = output['1'].reshape(-1, 60)

        t_loss = self.criterion_huber(J_t, J_t_gt)
        R_loss = self.criterion_huber(J_R_exp_rel, R_rel_gt)
        cls_loss = self.criterion_cls(R_label_scores, R_label_gt.reshape(-1))

        loss = (
            t_loss + R_loss + cls_loss
        )

        return (
            loss,
            J_R,
            J_R_gt,
            J_t,
            J_t_gt
        )

    def training_step(self, batch: Any, batch_idx: int):
        (
            loss,
            transform,
            res_rot,
            skeleton_pos,
            res_pos,
        ) = self.step(batch)

        # log train metrics
        joint_pos_error = self.train_jpe(skeleton_pos, res_pos)
        joint_ori_error = self.train_joe(transform, res_rot)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "train/jpe", joint_pos_error, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "train/joe", joint_ori_error, on_step=False, on_epoch=True, prog_bar=False
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_jpe.reset()
        self.train_joe.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        (
            loss,
            transform,
            res_rot,
            skeleton_pos,
            res_pos,
        ) = self.step(batch)
        batch_size = transform.shape[0]

        # log val metrics
        joint_pos_error = self.val_jpe(skeleton_pos, res_pos)
        joint_ori_error = self.val_joe(transform, res_rot)
        self.val_err.update(loss, weight=batch_size)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "val/jpe", joint_pos_error, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val/joe", joint_ori_error, on_step=False, on_epoch=True, prog_bar=True
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

    def test_step(self, batch: Any, batch_idx: int):
        (
            loss,
            transform,
            res_rot,
            skeleton_pos,
            res_pos,
        ) = self.step(batch)

        # log test metrics
        joint_pos_error = self.test_jpe(skeleton_pos, res_pos)
        joint_ori_error = self.test_joe(transform, res_rot)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/jpe", joint_pos_error, on_step=False, on_epoch=True)
        self.log("test/joe", joint_ori_error, on_step=False, on_epoch=True)

        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        self.test_jpe.reset()
        self.test_joe.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.hparams.lr_decay)
        return [optimizer], [scheduler]
