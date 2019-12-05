import torch.nn as nn
import torch
from mmdet.models.registry import HEADS
from mmdet.core import force_fp32
from ..builder import build_loss


@HEADS.register_module
class FCTranslationHead(nn.Module):
    r"""More general bbox head, with shared fc (bboxes) and fc (carclsrot) layers and two optional
    separated branches.

    car cls rot fcs ->
                                    -> (Addition) translation reg -> translation
    bboxes fcs -> bboxes fcs ->

    """

    def __init__(self,
                 in_channels_bboxes=4,
                 in_channels_carclsrot=1024,
                 fc_out_channels=100,
                 num_translation_reg=3,
                 loss_translation=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 *args, **kwargs):
        super(FCTranslationHead, self).__init__(*args, **kwargs)

        self.in_channels_bboxes = in_channels_bboxes
        self.in_channels_carclsrot = in_channels_carclsrot
        self.num_translation_reg = num_translation_reg

        self.car_cls_rot_linear = nn.Linear(in_channels_carclsrot, fc_out_channels)
        self.bboxes_linear_1 = nn.Linear(in_channels_bboxes, fc_out_channels)
        self.bboxes_linear_2 = nn.Linear(fc_out_channels, fc_out_channels)
        self.trans_pred = nn.Linear(fc_out_channels + fc_out_channels, num_translation_reg)
        self.relu = nn.ReLU(inplace=True)
        # Di Wu add build loss here overriding bbox_head
        self.loss_translation = build_loss(loss_translation)

        # camera intrinsic also is saved here:
        self.fx, self.cx, self.fy, self.cy = 2304.5479, 1686.2379, 2305.8757, (2710 - 1480)/2
        # translation mean and std:
        self.t_x_mean, self.t_y_mean, self.t_z_mean = -3, 9, 50
        self.t_x_std,  self.t_y_std, self.t_z_std = 14.015, 4.695, 29.596

    def init_weights(self):
        super(FCTranslationHead, self).init_weights()
        for module_list in [self.bboxes_linear_1, self.bboxes_linear_2, self.car_cls_rot_linear,
                            self.trans_pred]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_mlp, x_car_cls_rot=None):
        # shared part
        x_bbox_feat = self.relu(self.bboxes_linear_1(x_mlp))
        x_bbox_feat = self.relu(self.bboxes_linear_2(x_bbox_feat))

        x_carclsrot_feat = self.relu(self.car_cls_rot_linear(x_car_cls_rot))

        x_merge = self.relu(torch.cat((x_bbox_feat, x_carclsrot_feat), dim=1))

        trans_pred = self.trans_pred(x_merge)
        return trans_pred

    def get_target(self, sampling_results, rcnn_train_cfg=None):

        pos_gt_assigned_translations = [res.pos_gt_assigned_translations for res in sampling_results]
        pos_gt_assigned_translations = torch.cat(pos_gt_assigned_translations, 0)

        pos_gt_assigned_translations[:, 0] -= self.t_x_mean
        pos_gt_assigned_translations[:, 0] /= self.t_x_std
        pos_gt_assigned_translations[:, 1] -= self.t_y_mean
        pos_gt_assigned_translations[:, 1] /= self.t_y_std
        pos_gt_assigned_translations[:, 2] -= self.t_z_mean
        pos_gt_assigned_translations[:, 2] /= self.t_z_std

        return pos_gt_assigned_translations

    def bbox_transform_pytorch(self, rois, scale_factor, device_id):
        """Forward transform that maps proposal boxes to predicted ground-truth
        boxes using bounding-box regression deltas. See bbox_transform_inv for a
        description of the weights argument.
        This is a pytorch head
        """

        rois = rois / scale_factor  # We transform it back to the original pixel space before resizing
        widths = rois[:, 2] - rois[:, 0]
        heights = rois[:, 3] - rois[:, 1]
        ctr_x = rois[:, 0] + 0.5 * widths
        ctr_y = rois[:, 1] + 0.5 * heights

        pred_boxes = torch.zeros(rois.shape, dtype=rois.dtype).cuda(device_id)

        pred_boxes[:, 0] = ctr_x
        pred_boxes[:, 1] = ctr_y
        pred_boxes[:, 2] = widths
        pred_boxes[:, 3] = heights

        pred_boxes[:, 0] -= self.cx
        pred_boxes[:, 0] /= self.fx
        pred_boxes[:, 1] -= self.cy
        pred_boxes[:, 1] /= self.fy

        pred_boxes[:, 2] /= self.fx
        pred_boxes[:, 3] /= self.fy

        return pred_boxes

    @force_fp32(apply_to=('translation_pred'))
    def loss(self,
             translation_pred,
             translation_target):

        losses = dict()
        losses['loss_translation'] = self.translation_distance(translation_pred, translation_target)
        losses['translation_distance'] = self.translation_distance(translation_pred, translation_target)

        return losses

    def translation_distance(self, translation_pred, translation_target):
        diff = translation_pred - translation_target

        diff[:, 0] *= self.t_x_std
        diff[:, 1] *= self.t_y_std
        diff[:, 2] *= self.t_z_std

        translation_diff = torch.mean(torch.sqrt(torch.sum(diff**2, dim=1)))
        return translation_diff

    def pred_to_world_coord(self, translation_pred):

        translation_pred[:, 0] *= self.t_x_std
        translation_pred[:, 0] += self.t_x_mean

        translation_pred[:, 1] *= self.t_y_std
        translation_pred[:, 1] += self.t_y_mean

        translation_pred[:, 2] *= self.t_z_std
        translation_pred[:, 2] += self.t_z_mean

        return translation_pred

@HEADS.register_module
class SharedTranslationHead(FCTranslationHead):

    def __init__(self, *args, **kwargs):
        super(SharedTranslationHead, self).__init__(
            # in_channels_bboxes=4,
            # in_channels_carclsrot=1024,
            # fc_out_channels=100,
            # num_translation_reg=3,
            # loss_translation=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
            *args,
            **kwargs)
