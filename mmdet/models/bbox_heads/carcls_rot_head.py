import torch.nn as nn
import torch
import numpy as np

from mmdet.models.registry import HEADS
from mmdet.models.utils import ConvModule
from .bbox_head import BBoxHead
from mmdet.core import force_fp32

from ..losses import accuracy
from ..builder import build_loss


@HEADS.register_module
class ConvFCCarClsRotHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_car_cls='CrossEntropyLoss',
                 loss_quaternion='L1',
                 *args,
                 **kwargs):
        super(ConvFCCarClsRotHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                                                             self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

        # Di Wu add build loss here overriding bbox_head
        self.loss_car_cls = build_loss(loss_car_cls)
        if loss_quaternion['type'] == 'L1':
            self.loss_quaternion = nn.L1Loss()
        else:
            self.loss_quaternion = build_loss(loss_quaternion)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCCarClsRotHead, self).init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, res_feat=None, return_logits=False, return_feat=False, return_last=False):
        """

        :param x:
        :param res_feat:
        :param return_logits:
        :param return_feat: used for interleaving (currently no use!)
        :param return_last:  used for tranlsation estimation
        :return:
        """

        # Currently, res_feat has no use!
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x
        last_feat = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        car_cls_score_pred = self.fc_cls(x_cls) if self.with_cls else None
        quaternion_pred = self.fc_reg(x_reg) if self.with_reg else None
        # Di WU also normalise the quaternion here
        quaternion_pred = nn.functional.normalize(quaternion_pred, p=2, dim=1)

        outs = []
        if return_logits:
            outs.append(car_cls_score_pred)
            outs.append(quaternion_pred)
        if return_feat:
            raise NotImplementedError
            # outs.append(res_feat)
        if return_last:
            outs.append(last_feat)
        return outs if len(outs) > 1 else outs[0]

    def get_target(self, sampling_results, carlabels, quaternion_semispheres,
                   rcnn_train_cfg):

        pos_carlabels = [res.pos_gt_carlabels for res in sampling_results]
        pos_gt_assigned_quaternion_semispheres = [res.pos_gt_assigned_quaternion_semispheres for res in
                                                  sampling_results]
        # pog_gt_assigned_translations = [res.pog_gt_assigned_translations for res in sampling_results]
        pos_carlabels = torch.cat(pos_carlabels, 0)
        pos_gt_assigned_quaternion_semispheres = torch.cat(pos_gt_assigned_quaternion_semispheres, 0)

        return pos_carlabels, pos_gt_assigned_quaternion_semispheres

    @force_fp32(apply_to=('car_cls_score', 'quaternion_pred'))
    def loss(self,
             car_cls_score_pred,
             quaternion_pred,
             car_cls_score_target,
             quaternion_target):
        losses = dict()

        losses['car_cls_ce_loss'] = self.loss_car_cls(car_cls_score_pred, car_cls_score_target)
        losses['car_cls_acc'] = accuracy(car_cls_score_pred, car_cls_score_target)

        losses['loss_quaternion'] = self.loss_quaternion(quaternion_pred, quaternion_target)
        losses['rotation_distance'] = self.rotation_similiarity(quaternion_pred, quaternion_target)

        return losses

    def rotation_similiarity(self, quaternion_pred, quaternion_target):
        diff = torch.abs(torch.sum(quaternion_pred * quaternion_target, dim=1))
        dis_rot = torch.mean(2 * torch.acos(diff) * 180 / np.pi)
        return dis_rot.detach()


@HEADS.register_module
class SharedCarClsRotHead(ConvFCCarClsRotHead):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(SharedCarClsRotHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
