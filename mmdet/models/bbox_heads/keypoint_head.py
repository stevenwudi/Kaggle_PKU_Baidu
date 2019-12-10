import torch.nn as nn
import torch
from mmdet.models.registry import HEADS
from mmdet.core import force_fp32
from ..builder import build_loss


@HEADS.register_module
class FCKeyPointHead(nn.Module):
    r"""More general bbox head, with shared fc (bboxes) and fc (carclsrot) layers and two optional
    separated branches.

    car cls rot fcs ->
                                    -> (Addition) keypoint reg -> keypoint
    bboxes fcs -> bboxes fcs ->

    """

    def __init__(self,
                 in_channels_bboxes=4,
                 in_channels_keypoint=1024,
                 fc_out_channels=512,
                 num_keypoint_reg=132,
                 loss_keypoint=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 *args, **kwargs):
        super(FCKeyPointHead, self).__init__(*args, **kwargs)

        self.in_channels_bboxes = in_channels_bboxes
        self.in_channels_keypoint = in_channels_keypoint
        self.num_keypoint_reg = num_keypoint_reg

        self.keypoint_linear = nn.Linear(in_channels_keypoint, fc_out_channels)
        self.bboxes_linear_1 = nn.Linear(in_channels_bboxes, fc_out_channels)
        self.bboxes_linear_2 = nn.Linear(fc_out_channels, fc_out_channels)
        self.keypoint_pred = nn.Linear(fc_out_channels + fc_out_channels, num_keypoint_reg)
        self.relu = nn.ReLU(inplace=True)
        self.loss_keypoint = build_loss(loss_keypoint)

        # camera intrinsic also is saved here:
        self.fx, self.cx, self.fy, self.cy = 2304.5479, 1686.2379, 2305.8757, (2710 - 1480)/2

    def init_weights(self):
        super(FCKeyPointHead, self).init_weights()
        for module_list in [self.bboxes_linear_1, self.bboxes_linear_2, self.car_cls_rot_linear,
                            self.trans_pred]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_mlp, x_keypoint=None):
        # shared part
        x_bbox_feat = self.relu(self.bboxes_linear_1(x_mlp))
        x_bbox_feat = self.relu(self.bboxes_linear_2(x_bbox_feat))

        x_keypoint_feat = self.relu(self.keypoint_linear(x_keypoint))

        x_merge = self.relu(torch.cat((x_bbox_feat, x_keypoint_feat), dim=1))

        keypoint_pred = self.keypoint_pred(x_merge)
        return keypoint_pred

    def get_target(self, sampling_results, rcnn_train_cfg=None):

        pos_gt_assigned_keypoints = [res.pos_gt_assigned_keypoints for res in sampling_results]
        pos_gt_assigned_keypoints = torch.cat(pos_gt_assigned_keypoints, 0)

        return pos_gt_assigned_keypoints
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


    @force_fp32(apply_to=('keypoint_pred'))
    def loss(self,keypoint_pred,keypoint_target):

        losses = dict()
        losses['loss_keypoint'] = self.keypoint_distance(keypoint_pred, keypoint_target)

        return losses

    def keypoint_distance(self, keypoint_pred, keypoint_target):
        diff = keypoint_pred - keypoint_target
        keypoint_diff = torch.mean(torch.sqrt(torch.sum(diff**2, dim=1)))
        return keypoint_diff

@HEADS.register_module
class SharedKeyPointHead(FCKeyPointHead):

    def __init__(self, *args, **kwargs):
        super(SharedKeyPointHead, self).__init__(
            # in_channels_bboxes=4,
            # in_channels_carclsrot=1024,
            # fc_out_channels=100,
            # num_translation_reg=3,
            # loss_translation=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
            *args,
            **kwargs)
