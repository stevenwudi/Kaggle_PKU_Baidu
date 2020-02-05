import torch.nn as nn
import numpy as np
import torch
from mmdet.models.registry import HEADS
from mmdet.core import force_fp32
import mmcv
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
                 bbox_relative=False,  # if bbox_relative=False, then it requires training/test input the same
                 translation_bboxes_regression=False,
                 bboxes_regression=dict(type='maxIoU', iou_thresh=0.1),  #
                 loss_translation=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 *args, **kwargs):
        super(FCTranslationHead, self).__init__(*args, **kwargs)

        self.in_channels_bboxes = in_channels_bboxes
        self.in_channels_carclsrot = in_channels_carclsrot
        self.num_translation_reg = num_translation_reg
        self.bbox_relative = bbox_relative
        self.car_cls_rot_linear = nn.Linear(in_channels_carclsrot, fc_out_channels)
        self.bboxes_linear_1 = nn.Linear(in_channels_bboxes, fc_out_channels)
        self.bboxes_linear_2 = nn.Linear(fc_out_channels, fc_out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Di Wu add build loss here overriding bbox_head
        self.loss_translation = build_loss(loss_translation)
        self.bboxes_regression = bboxes_regression

        # if we use bboxes to regress the x,y,z
        self.translation_bboxes_regression = translation_bboxes_regression
        if self.translation_bboxes_regression:
            bboxes_file_name = '../mmdet/models/bbox_heads/bboxes_with_translation_pick_543.pkl'
            try:
                self.bboxes_with_translation_pick = mmcv.load(bboxes_file_name)
                print('Finish loading file: %s' % bboxes_file_name)
                # The translational prediction will now be dependend upon anchor boxes
                num_anchor_boxes = self.bboxes_with_translation_pick.shape[0]
                self.trans_pred = nn.Linear(fc_out_channels + fc_out_channels, num_anchor_boxes * num_translation_reg)

            except IOError:
                print('There was an error opening the file!')
                return
        else:
            self.trans_pred = nn.Linear(fc_out_channels + fc_out_channels, num_translation_reg)

        # camera intrinsic also is saved here:
        self.fx, self.cx, self.fy, self.cy = 2304.5479, 1686.2379, 2305.8757, (2710 - 1480) / 2
        # translation mean and std:
        self.t_x_mean, self.t_y_mean, self.t_z_mean = -3, 9, 50
        self.t_x_std, self.t_y_std, self.t_z_std = 14.015, 4.695, 29.596

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

    def get_target_trans_box(self,
                             sampling_results,
                             trans_pred,
                             rois_resize,
                             scale_factor,
                             device_id,
                             iou_thresh=0.5,
                             beta=0.1,):
        # ground truth
        pos_gt_assigned_translations = [res.pos_gt_assigned_translations for res in sampling_results]
        pos_gt_assigned_translations = torch.cat(pos_gt_assigned_translations, 0)

        rois = rois_resize / scale_factor  # We transform it back to the original pixel space before resizing
        rois = rois.cpu().data.numpy()
        # Now we find the IoU > iou_thresh
        boxes = self.bboxes_with_translation_pick.copy()
        # Because we crop the bottom
        boxes[:, 1] -= 1480
        boxes[:, 3] -= 1480

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        # get the real world coordinate [x, y, z]
        # pred_boxes denote the world coornidates of the referenced boxes
        boxes_world_xyz = torch.from_numpy(boxes[:, 4:].astype(rois.dtype)).cuda(device_id)
        distance = torch.sqrt(torch.sum(boxes_world_xyz ** 2, dim=1))

        losses = dict()
        losses['loss_translation'] = 0
        losses['translation_distance'] = 0
        losses['translation_distance_relative'] = 0
        for i, roi in enumerate(rois):
            area_roi = (roi[2] - roi[0] + 1) * (roi[3] - roi[1] + 1)
            xx1 = np.maximum(roi[0], x1)
            xx2 = np.minimum(roi[2], x2)
            yy1 = np.maximum(roi[1], y1)
            yy2 = np.minimum(roi[3], y2)
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / (area + area_roi - w * h)
            idx_overlap = overlap > iou_thresh
            # We follow SSD, find the max and other criterion
            idx_max = np.argmax(overlap)
            idx_overlap[idx_max] = True

            matched_idx = torch.Tensor(idx_overlap) == True
            matched_expand = matched_idx[:, None].expand(matched_idx.shape[0], 3).contiguous().view(-1)
            matched_expand = matched_expand.float().cuda(device_id)
            # calculate the reference g as in SSD paper eq. (2)
            g = (pos_gt_assigned_translations[i] - boxes_world_xyz) / distance[:, None]

            target_translations = g.view(-1) * matched_expand
            trans_pred[i] = trans_pred[i] * matched_expand

            # We still have smooth L1 loss with beta=0.1, because the translation threshold starts from 0.1
            diff = torch.abs(trans_pred[i] - target_translations)
            loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
            losses['loss_translation'] += loss.sum() / matched_expand.sum()

            # Get the world coordinate and distance
            translation_pred = self.get_trans_by_SSD_regression(trans_pred[i], boxes_world_xyz, distance, idx_max)
            translation_target = pos_gt_assigned_translations[i]
            diff_distance = translation_pred - translation_target
            distance_i = torch.sqrt(torch.sum(translation_target ** 2))
            translation_world = torch.sqrt(torch.sum(diff_distance ** 2))
            translation_world_relative = translation_world / distance_i
            losses['translation_distance'] += translation_world
            losses['translation_distance_relative'] += translation_world_relative

        # We still need to devide loss by the car number in an image
        losses['loss_translation'] /= len(rois)
        losses['translation_distance'] /= len(rois)
        losses['translation_distance_relative'] /= len(rois)
        # The metrics are detached from backpropagation
        losses['translation_distance'] = losses['translation_distance'].detach()
        losses['translation_distance_relative'] = losses['translation_distance_relative'].detach()

        return losses

    def get_trans_by_SSD_regression(self, trans_pred_i, boxes_world_xyz, distance, idx_max):

        trans_pred_i = trans_pred_i.view(boxes_world_xyz.shape[0], boxes_world_xyz.shape[1])
        real_world_coord = trans_pred_i * distance[:, None] + boxes_world_xyz

        return real_world_coord[idx_max]

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

    def bbox_transform_pytorch_relative(self, rois, scale_factor, device_id, ori_shape):
        """Forward transform that maps proposal boxes to predicted ground-truth
        boxes using bounding-box regression deltas. See bbox_transform_inv for a
        description of the weights argument.
        This is a pytorch head
        """
        ### TODO delete the following 3 lines #####
        # pad_shape = [1230, 3384]
        # pad_shape[0] *= 1248/576
        # pad_shape[1] *= 3392/1600
        #####

        rois = rois / scale_factor  # We transform invidiat back to the original pixel space (1280, 3384) before resizing
        widths = rois[:, 2] - rois[:, 0]
        heights = rois[:, 3] - rois[:, 1]
        ctr_x = rois[:, 0] + 0.5 * widths
        ctr_y = rois[:, 1] + 0.5 * heights

        pred_boxes = torch.zeros(rois.shape, dtype=rois.dtype).cuda(device_id)

        pred_boxes[:, 0] = ctr_x
        pred_boxes[:, 1] = ctr_y
        pred_boxes[:, 2] = widths
        pred_boxes[:, 3] = heights

        pred_boxes[:, 0] -= ori_shape[1] / 2
        pred_boxes[:, 0] /= ori_shape[1]
        pred_boxes[:, 1] -= ori_shape[0] / 2
        pred_boxes[:, 1] /= ori_shape[0]

        pred_boxes[:, 2] /= ori_shape[1]
        pred_boxes[:, 3] /= ori_shape[0]

        return pred_boxes

    @force_fp32(apply_to=('translation_pred'))
    def loss(self,
             translation_pred,
             translation_target):

        losses = dict()
        losses['loss_translation'] = self.translation_distance(translation_pred, translation_target)
        losses['translation_distance'] = self.translation_distance(translation_pred, translation_target)
        losses['translation_distance_relative'] = self.translation_distance_relative(translation_pred,
                                                                                     translation_target)
        # The metrics are detached from backpropagation
        losses['translation_distance'] = losses['translation_distance'].detach()
        losses['translation_distance_relative'] = losses['translation_distance_relative'].detach()
        return losses

    def translation_distance_relative(self, translation_pred, translation_target):
        diff = translation_pred - translation_target
        distance = torch.sqrt(torch.sum(translation_target ** 2, dim=1))
        translation_diff = torch.sqrt(torch.sum(diff ** 2, dim=1)) / distance
        return torch.mean(translation_diff)

    def translation_distance(self, translation_pred, translation_target):
        diff = translation_pred - translation_target

        diff[:, 0] *= self.t_x_std
        diff[:, 1] *= self.t_y_std
        diff[:, 2] *= self.t_z_std

        translation_diff = torch.mean(torch.sqrt(torch.sum(diff ** 2, dim=1)))
        return translation_diff

    def pred_to_world_coord(self, translation_pred):

        translation_pred[:, 0] *= self.t_x_std
        translation_pred[:, 0] += self.t_x_mean

        translation_pred[:, 1] *= self.t_y_std
        translation_pred[:, 1] += self.t_y_mean

        translation_pred[:, 2] *= self.t_z_std
        translation_pred[:, 2] += self.t_z_mean

        return translation_pred

    def pred_to_world_coord_SSD(self,
                                trans_pred,
                                rois_resize,
                                scale_factor,
                                device_id):

        rois = rois_resize / scale_factor  # We transform it back to the original pixel space before resizing
        rois = rois.cpu().data.numpy()

        # Now we find the IoU > iou_thresh
        boxes = self.bboxes_with_translation_pick.copy()
        # Because we crop the bottom
        boxes[:, 1] -= 1480
        boxes[:, 3] -= 1480

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        # get the real world coordinate [x, y, z]
        boxes_world_xyz = torch.from_numpy(boxes[:, 4:].astype(rois.dtype)).cuda(device_id)
        distance = torch.sqrt(torch.sum(boxes_world_xyz ** 2, dim=1))

        translation_pred = torch.zeros((rois.shape[0], 3), dtype=trans_pred.dtype).cuda(device_id)
        for i, roi in enumerate(rois):
            area_roi = (roi[2] - roi[0] + 1) * (roi[3] - roi[1] + 1)
            xx1 = np.maximum(roi[0], x1)
            xx2 = np.minimum(roi[2], x2)
            yy1 = np.maximum(roi[1], y1)
            yy2 = np.minimum(roi[3], y2)
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / (area + area_roi - w * h)
            # We follow SSD, find the max and other criterion
            idx_max = np.argmax(overlap)
            if self.bboxes_regression['type'] == 'maxIoU':
                # Get the world coordinate and distance use only the max IoU
                translation_pred[i] = self.get_trans_by_SSD_regression(trans_pred[i], boxes_world_xyz, distance, idx_max)
            elif self.bboxes_regression['type'] == 'allIoU':
                # Get the world coordinate from the bboxes that has IoU larger then a threshold
                idx_overlap = np.where(overlap > self.bboxes_regression['iou_thresh'])
                idx_overlap = np.union1d(idx_max, idx_overlap)
                translation_pred_list = [self.get_trans_by_SSD_regression(trans_pred[i], boxes_world_xyz, distance, idx) for idx in idx_overlap]
                translation_pred[i] = torch.stack(translation_pred_list).mean(dim=0)
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
