import torch
import torch.nn.functional as F
from torch import nn

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from .. import builder
from ..registry import DETECTORS
from .cascade_rcnn import CascadeRCNN


@DETECTORS.register_module
class HybridTaskCascade(CascadeRCNN):

    def __init__(self,
                 num_stages,
                 backbone,
                 semantic_roi_extractor=None,
                 semantic_head=None,
                 semantic_fusion=('bbox', 'mask'),
                 interleaved=True,
                 mask_info_flow=True,
                 car_cls_info_flow=True,
                 with_semantic_loss=False,
                 with_car_cls_rot=False,
                 with_translation=True,
                 **kwargs):
        super(HybridTaskCascade, self).__init__(num_stages, backbone, **kwargs)
        assert self.with_bbox and self.with_mask
        assert not self.with_shared_head  # shared head not supported
        if semantic_head is not None:
            self.semantic_roi_extractor = builder.build_roi_extractor(
                semantic_roi_extractor)
            self.semantic_head = builder.build_head(semantic_head)

        self.with_semantic_loss = with_semantic_loss
        self.semantic_fusion = semantic_fusion
        self.interleaved = interleaved
        self.mask_info_flow = mask_info_flow
        self.car_cls_info_flow = car_cls_info_flow

        # The following is for 6DoF estimation
        self.with_car_cls_rot = with_car_cls_rot
        self.with_translation = with_translation

        # Bayesian learning of the weight
        if self.bayesian_weight_learning and self.train_cfg is not None:
            self.fc_car_cls_weight = nn.Linear(in_features=1, out_features=1, bias=False)
            self.fc_rot_weight = nn.Linear(in_features=1, out_features=1, bias=False)
            self.fc_translation_weight = nn.Linear(in_features=1, out_features=1, bias=False)
            # initialise the weight here
            # https://discuss.pytorch.org/t/initialize-nn-linear-with-specific-weights/29005/2
            with torch.no_grad():
                self.fc_car_cls_weight.weight.copy_(torch.tensor(self.train_cfg.car_cls_weight))
                self.fc_rot_weight.weight.copy_(torch.tensor(self.train_cfg.rot_weight))
                self.fc_translation_weight.weight.copy_(torch.tensor(self.train_cfg.translation_weight))

    @property
    def with_semantic(self):
        if hasattr(self, 'semantic_head') and self.semantic_head is not None:
            return True
        else:
            return False

    def _bbox_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_bboxes,
                            gt_labels,
                            rcnn_train_cfg,
                            semantic_feat=None):
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        # semantic feature fusion
        # element-wise sum for original features and pooled semantic features
        if self.with_semantic and 'bbox' in self.semantic_fusion:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = F.adaptive_avg_pool2d(
                    bbox_semantic_feat, bbox_feats.shape[-2:])
            bbox_feats += bbox_semantic_feat

        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_targets = bbox_head.get_target(sampling_results, gt_bboxes,
                                            gt_labels, rcnn_train_cfg)
        loss_bbox = bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
        return loss_bbox, rois, bbox_targets, bbox_pred

    def _carcls_rot_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            carlabels,
                            quaternion_semispheres,
                            rcnn_train_cfg,
                            semantic_feat=None):
        car_cls_rot_roi_extractor = self.car_cls_rot_roi_extractor[stage]
        car_cls_rot_head = self.car_cls_rot_head[stage]

        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        car_cls_rot_feats = car_cls_rot_roi_extractor(x[:car_cls_rot_roi_extractor.num_inputs], pos_rois)

        # semantic feature fusion
        # element-wise sum for original features and pooled semantic features
        if self.with_semantic and 'car_cls_rot' in self.semantic_fusion:
            car_cls_rot_semantic_feat = self.semantic_roi_extractor([semantic_feat], pos_rois)
            if car_cls_rot_semantic_feat.shape[-2:] != car_cls_rot_feats.shape[-2:]:
                car_cls_rot_semantic_feat = F.adaptive_avg_pool2d(
                    car_cls_rot_semantic_feat, car_cls_rot_feats.shape[-2:])
            car_cls_rot_feats += car_cls_rot_semantic_feat

        # car cls rot information flow
        # forward all previous mask heads to obtain last_feat, and fuse it
        # with the normal mask feature
        if self.car_cls_info_flow:
            raise NotImplementedError
            last_feat = None
            # for i in range(stage):
            #     last_feat = self.car_cls_rot_head[i](car_cls_rot_feats, last_feat, return_logits=False)
            last_feat, car_cls_rot_feat = car_cls_rot_head(car_cls_rot_feats, last_feat, return_logits=False, return_feat=False, return_last=False)
        else:
            car_cls_score_pred, quaternion_pred, car_cls_rot_feat = car_cls_rot_head(car_cls_rot_feats, return_logits=True, return_last=True)

        car_cls_score_target, quaternion_target = car_cls_rot_head.get_target(sampling_results, carlabels, quaternion_semispheres, rcnn_train_cfg)

        loss_car_cls_rot = car_cls_rot_head.loss(car_cls_score_pred, quaternion_pred,
                                                 car_cls_score_target, quaternion_target)
        return loss_car_cls_rot, car_cls_rot_feat

    def _translation_forward_train(self, sampling_results, scale_factor, car_cls_rot_feat, img_meta):
        pos_bboxes = [res.pos_bboxes for res in sampling_results]
        # TODO: this is a dangerous hack: we assume only one image per batch
        if len(pos_bboxes) > 1:
            raise NotImplementedError("Image batch size 1 is not implement!")
        for im_idx in range(len(pos_bboxes)):
            device_id = car_cls_rot_feat.get_device()
            if self.translation_head.bbox_relative:
                ori_shape = img_meta[im_idx]['ori_shape']
                # then we use relative information instead the absolute world space
                pred_boxes = self.translation_head.bbox_transform_pytorch_relative(pos_bboxes[im_idx], scale_factor[im_idx], device_id, ori_shape)
            else:
                pred_boxes = self.translation_head.bbox_transform_pytorch(pos_bboxes[im_idx], scale_factor[im_idx], device_id)
            trans_pred = self.translation_head(pred_boxes, car_cls_rot_feat)

            if self.translation_head.translation_bboxes_regression:
                loss_translation = self.translation_head.get_target_trans_box(sampling_results, trans_pred,
                                                                              pos_bboxes[im_idx], scale_factor[im_idx],
                                                                              device_id)
            else:
                pos_gt_assigned_translations = self.translation_head.get_target(sampling_results)
                loss_translation = self.translation_head.loss(trans_pred, pos_gt_assigned_translations)

        return loss_translation

    def _mask_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            semantic_feat=None):
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        pos_rois)

        # semantic feature fusion
        # element-wise sum for original features and pooled semantic features
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             pos_rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats += mask_semantic_feat

        # mask information flow
        # forward all previous mask heads to obtain last_feat, and fuse it
        # with the normal mask feature
        if self.mask_info_flow:
            last_feat = None
            for i in range(stage):
                last_feat = self.mask_head[i](
                    mask_feats, last_feat, return_logits=False)
            mask_pred = mask_head(mask_feats, last_feat, return_feat=False)
        else:
            mask_pred = mask_head(mask_feats)

        mask_targets = mask_head.get_target(sampling_results, gt_masks,
                                            rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)
        return loss_mask

    def _bbox_forward_test(self, stage, x, rois, semantic_feat=None):
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(
            x[:len(bbox_roi_extractor.featmap_strides)], rois)
        if self.with_semantic and 'bbox' in self.semantic_fusion:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = F.adaptive_avg_pool2d(
                    bbox_semantic_feat, bbox_feats.shape[-2:])
            bbox_feats += bbox_semantic_feat
        cls_score, bbox_pred = bbox_head(bbox_feats)
        return cls_score, bbox_pred

    def _mask_forward_test(self, stage, x, bboxes, semantic_feat=None):
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_rois = bbox2roi([bboxes])
        mask_feats = mask_roi_extractor(
            x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             mask_rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats += mask_semantic_feat
        if self.mask_info_flow:
            last_feat = None
            last_pred = None
            for i in range(stage):
                mask_pred, last_feat = self.mask_head[i](mask_feats, last_feat)
                if last_pred is not None:
                    mask_pred = mask_pred + last_pred
                last_pred = mask_pred
            mask_pred = mask_head(mask_feats, last_feat, return_feat=False)
            if last_pred is not None:
                mask_pred = mask_pred + last_pred
        else:
            mask_pred = mask_head(mask_feats)
        return mask_pred

    def _carcls_rot_forward_test(self,
                              stage,
                              x,
                              _bboxes,
                              semantic_feat=None):
        # We have only one extractor
        car_cls_rot_roi_extractor = self.car_cls_rot_roi_extractor[-1]
        ## Another bug here between training and inferencing
        pos_rois = bbox2roi([_bboxes])
        # pos_rois_shift = pos_rois
        # pos_rois_shift[:, 1:] = pos_rois[:, :4]
        car_cls_rot_feats = car_cls_rot_roi_extractor(x[:car_cls_rot_roi_extractor.num_inputs], pos_rois)

        if self.car_cls_info_flow:
            raise NotImplementedError
            # The following code does not exist
            # last_feat = None
            # for i in range(self.num_stages):
            #     car_cls_rot_head = self.car_cls_rot_head[i]
            #     if self.car_cls_info_flow:
            #         last_feat = self.car_cls_rot_head[i](car_cls_rot_feats, last_feat, return_logits=False)

        # No information flow yet
        car_cls_rot_head = self.car_cls_rot_head[-1]
        car_cls_score_pred, quaternion_pred, car_cls_rot_feat = car_cls_rot_head(car_cls_rot_feats, return_logits=True, return_last=True)
        car_cls_score_pred = car_cls_score_pred.cpu().numpy()
        quaternion_pred = quaternion_pred.cpu().numpy()
        return car_cls_score_pred, quaternion_pred, car_cls_rot_feat

    def _translation_forward_test(self, pos_bboxes, scale_factor, car_cls_rot_feat, ori_shape):

        # TODO: this is a dangerous hack: we assume only one image per batch
        device_id = car_cls_rot_feat.get_device()

        if self.translation_head.bbox_relative:
            # then we use relative information instead the absolute world space
            pred_boxes = self.translation_head.bbox_transform_pytorch_relative(pos_bboxes, scale_factor, device_id, ori_shape)
        else:
            pred_boxes = self.translation_head.bbox_transform_pytorch(pos_bboxes, scale_factor, device_id)
        trans_pred = self.translation_head(pred_boxes, car_cls_rot_feat)
        if self.translation_head.translation_bboxes_regression:
            trans_pred_world = self.translation_head.pred_to_world_coord_SSD(trans_pred, pos_bboxes, scale_factor, device_id)
        else:
            trans_pred_world = self.translation_head.pred_to_world_coord(trans_pred)
        trans_pred_world = trans_pred_world.cpu().numpy()

        return trans_pred_world

    def forward_dummy(self, img):
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).cuda()
        # semantic head
        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None
        # bbox heads
        rois = bbox2roi([proposals])
        for i in range(self.num_stages):
            cls_score, bbox_pred = self._bbox_forward_test(
                i, x, rois, semantic_feat=semantic_feat)
            outs = outs + (cls_score, bbox_pred)
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            mask_roi_extractor = self.mask_roi_extractor[-1]
            mask_feats = mask_roi_extractor(
                x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_semantic and 'mask' in self.semantic_fusion:
                mask_semantic_feat = self.semantic_roi_extractor(
                    [semantic_feat], mask_rois)
                mask_feats += mask_semantic_feat
            last_feat = None
            for i in range(self.num_stages):
                mask_head = self.mask_head[i]
                if self.mask_info_flow:
                    mask_pred, last_feat = mask_head(mask_feats, last_feat)
                else:
                    mask_pred = mask_head(mask_feats)
                outs = outs + (mask_pred, )

        # car cls and rot head
        if self.with_car_cls_rot:
            pos_rois = rois[:100]
            car_cls_rot_roi_extractor = self.car_cls_rot_roi_extractor[-1]
            car_cls_rot_feats = car_cls_rot_roi_extractor(x[:car_cls_rot_roi_extractor.num_inputs], pos_rois)
            for i in range(self.num_stages):
                car_cls_rot_head = self.car_cls_rot_head[i]
                if self.car_cls_info_flow:
                    last_feat = None
                    for ii in range(i):
                        last_feat = self.car_cls_rot_head[ii](car_cls_rot_feats, last_feat, return_logits=False)
                    car_cls_score_pred, quaternion_pred = car_cls_rot_head(car_cls_rot_feats, last_feat, return_feat=False)
                else:
                    car_cls_score_pred, quaternion_pred = car_cls_rot_head(car_cls_rot_feats)

                outs = outs + (car_cls_score_pred, quaternion_pred)
        return outs

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None,
                      proposals=None,
                      carlabels=None,
                      quaternion_semispheres=None,
                      translations=None,
                      scale_factor=1.0,
                      ):
        x = self.extract_feat(img)

        losses = dict()
        # RPN part, the same as normal two-stage detectors
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)
            proposal_cfg = self.train_cfg.get('rpn_proposal',self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # semantic segmentation part
        # 2 outputs: segmentation prediction and embedded features
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
            if self.with_semantic_loss:
                loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_seg)
                losses['loss_semantic_seg'] = loss_seg
        else:
            semantic_feat = None

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg.rcnn[i]
            lw = self.train_cfg.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            bbox_assigner = build_assigner(rcnn_train_cfg.assigner)
            bbox_sampler = build_sampler(rcnn_train_cfg.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[j],
                                                     gt_bboxes[j],
                                                     gt_bboxes_ignore[j],
                                                     gt_labels[j],
                                                     carlabels[j],
                                                     quaternion_semispheres[j],
                                                     translations[j])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes[j],
                    gt_labels[j],
                    carlabels[j],
                    quaternion_semispheres[j],
                    translations[j],
                    feats=[lvl_feat[j][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            # bbox head forward and loss
            loss_bbox, rois, bbox_targets, bbox_pred = \
                self._bbox_forward_train(
                    i, x, sampling_results, gt_bboxes, gt_labels,
                    rcnn_train_cfg, semantic_feat)
            roi_labels = bbox_targets[0]

            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(i, name)] = (value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                # interleaved execution: use regressed bboxes by the box branch
                # to train the mask branch
                if self.interleaved:
                    pos_is_gts = [res.pos_is_gt for res in sampling_results]
                    with torch.no_grad():
                        proposal_list = self.bbox_head[i].refine_bboxes(
                            rois, roi_labels, bbox_pred, pos_is_gts, img_meta)
                        # re-assign and sample 512 RoIs from 512 RoIs
                        sampling_results = []
                        for j in range(num_imgs):
                            assign_result = bbox_assigner.assign(
                                proposal_list[j], gt_bboxes[j],
                                gt_bboxes_ignore[j], gt_labels[j],
                                carlabels[j],
                                quaternion_semispheres[j],
                                translations[j])
                            sampling_result = bbox_sampler.sample(
                                assign_result,
                                proposal_list[j],
                                gt_bboxes[j],
                                gt_labels[j],
                                carlabels[j],
                                quaternion_semispheres[j],
                                translations[j],
                                feats=[lvl_feat[j][None] for lvl_feat in x])
                            sampling_results.append(sampling_result)
                loss_mask = self._mask_forward_train(i, x, sampling_results,
                                                     gt_masks, rcnn_train_cfg,
                                                     semantic_feat)
                for name, value in loss_mask.items():
                    losses['s{}.{}'.format(i, name)] = (
                        value * lw if 'loss' in name else value)

            if self.with_car_cls_rot:

                loss_car_cls_rot, car_cls_rot_feat = self._carcls_rot_forward_train(i, x, sampling_results,
                                                                 carlabels, quaternion_semispheres,
                                                                 rcnn_train_cfg, semantic_feat)
                for name, value in loss_car_cls_rot.items():
                    losses['s{}.{}'.format(i, name)] = (value * lw if 'loss' in name else value)

            # refine bboxes (same as Cascade R-CNN)
            if i < self.num_stages - 1 and not self.interleaved:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        rois, roi_labels, bbox_pred, pos_is_gts, img_meta)

        # for translation, we don't have interleave or cascading for the moment
        if self.with_translation:
            loss_translation = self._translation_forward_train(sampling_results, scale_factor, car_cls_rot_feat, img_meta)
            for name, value in loss_translation.items():
                losses['s{}.{}'.format(i, name)] = (value * lw if 'loss' in name else value)

        # we change the dictionary key so that they plot in one row
        htc_keys = ['loss_rpn_cls', 'loss_rpn_bbox', 's0.loss_cls', 's0.acc', 's0.loss_bbox', 's0.loss_mask',
                    's1.loss_cls', 's1.acc', 's1.loss_bbox', 's1.loss_mask', 's2.loss_cls',  's2.acc', 's2.loss_bbox',
                    's2.loss_mask']

        kaggle_keys = ['s0.car_cls_ce_loss', 's0.car_cls_acc', 's0.loss_quaternion', 's0.rotation_distance',
                       's1.car_cls_ce_loss', 's1.car_cls_acc', 's1.loss_quaternion', 's1.rotation_distance',
                       's2.car_cls_ce_loss', 's2.car_cls_acc', 's2.loss_quaternion', 's2.rotation_distance',
                       's2.loss_translation', 's2.translation_distance', 's2.translation_distance_relative']
        for key in htc_keys:
            new_key = 'htc/' + key
            losses[new_key] = losses[key]
            del losses[key]
        for key in kaggle_keys:
            if key in losses.keys():
                new_key = 'kaggle/' + key
                losses[new_key] = losses[key]
                del losses[key]

        # if we use bayesian weight learning scheme as in:
        # Geometric loss functions for camera pose regression with deep learning
        # s = log (sigma) **2
        if self.bayesian_weight_learning:
            for key in losses.keys():
                if 'car_cls_ce_loss' in key:
                    losses[key] = self.fc_car_cls_weight(losses[key].expand(1)).squeeze()
                elif 'loss_quaternion' in key:
                    losses[key] = self.fc_rot_weight(losses[key].expand(1)).squeeze()
                elif 'loss_translation' in key:
                    losses[key] = self.fc_translation_weight(losses[key].expand(1)).squeeze()

            losses['weight/car_cls_weight_sigma_loss'] = - torch.log(self.fc_car_cls_weight.weight)
            losses['weight/rot_weight_sigma_loss'] = - torch.log(self.fc_rot_weight.weight)
            losses['weight/translation_weight_sigma_loss'] = - torch.log(self.fc_translation_weight.weight)

            # We just show the weight here, hence detach them from the computational graph
            losses['weight/car_cls_weight'] = self.fc_car_cls_weight.weight.detach()
            losses['weight/rot_weight'] = self.fc_rot_weight.weight.detach()
            losses['weight/translation_weight'] = self.fc_translation_weight.weight.detach()

        else:
            for key in losses.keys():
                if 'car_cls_ce_loss' in key:
                    losses[key] *= self.train_cfg.car_cls_weight
                elif 'loss_quaternion' in key:
                    losses[key] *= self.train_cfg.rot_weight
                elif 'loss_translation' in key:
                    losses[key] *= self.train_cfg.translation_weight

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None

        file_name = img_meta[0]['filename']
        img_shape = img_meta[0]['img_shape']
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_6dof_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg.rcnn

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_head = self.bbox_head[i]
            cls_score, bbox_pred = self._bbox_forward_test(i, x, rois, semantic_feat=semantic_feat)
            ms_scores.append(cls_score)

            if self.test_cfg.keep_all_stages:
                det_bboxes, det_labels = bbox_head.get_det_bboxes(
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    scale_factor,
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
                bbox_result = bbox2result(det_bboxes, det_labels,
                                          bbox_head.num_classes)
                ms_bbox_result['stage{}'.format(i)] = bbox_result

                if self.with_mask:
                    mask_head = self.mask_head[i]
                    if det_bboxes.shape[0] == 0:
                        mask_classes = mask_head.num_classes - 1
                        segm_result = [[] for _ in range(mask_classes)]
                    else:
                        _bboxes = (
                            det_bboxes[:, :4] *
                            scale_factor if rescale else det_bboxes)
                        mask_pred = self._mask_forward_test(
                            i, x, _bboxes, semantic_feat=semantic_feat)
                        segm_result = mask_head.get_seg_masks(
                            mask_pred, _bboxes, det_labels, rcnn_test_cfg,
                            ori_shape, scale_factor, rescale)
                    ms_segm_result['stage{}'.format(i)] = segm_result

            if i < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
                                                  img_meta[0])

        cls_score = sum(ms_scores) / float(len(ms_scores))
        det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                mask_classes = self.mask_head[-1].num_classes - 1
                segm_result = [[] for _ in range(mask_classes)]
            else:
                _bboxes = (
                    det_bboxes[:, :4] *
                    scale_factor if rescale else det_bboxes)

                mask_rois = bbox2roi([_bboxes])
                aug_masks = []
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    mask_feats += mask_semantic_feat
                last_feat = None
                for i in range(self.num_stages):
                    mask_head = self.mask_head[i]
                    if self.mask_info_flow:
                        mask_pred, last_feat = mask_head(mask_feats, last_feat)
                    else:
                        mask_pred = mask_head(mask_feats)
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                merged_masks = merge_aug_masks(aug_masks,
                                               [img_meta] * self.num_stages,
                                               self.test_cfg.rcnn)
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks, _bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, scale_factor, rescale)
            ms_segm_result['ensemble'] = segm_result

        if self.with_car_cls_rot:
            if self.test_cfg.keep_all_stages:
                raise NotImplementedError
            else:
                car_cls_coco = 2
                stage_num = self.num_stages-1
                pos_box = det_bboxes[det_labels == car_cls_coco]
                # !!!!!!!!!!!!!!!!!!!!! Quite import bug below, scale is needed!!!!!!!!!!!!!
                pos_box = (pos_box * scale_factor if rescale else det_bboxes)

                if len(pos_box):
                    car_cls_score_pred, quaternion_pred, car_cls_rot_feats = self._carcls_rot_forward_test(stage_num, x, pos_box, semantic_feat)
                else:
                    car_cls_score_pred, quaternion_pred, car_cls_rot_feats = [], [], []
            if self.with_translation:
                if len(pos_box):
                    trans_pred_world = self._translation_forward_test(pos_box[:, :4], scale_factor, car_cls_rot_feats, ori_shape)
                else:
                    trans_pred_world = []
            ms_6dof_result['ensemble'] = {'car_cls_score_pred': car_cls_score_pred,
                                          'quaternion_pred': quaternion_pred,
                                          'trans_pred_world': trans_pred_world,
                                          'file_name': file_name}
        if not self.test_cfg.keep_all_stages:
            if self.with_translation:
                results = (ms_bbox_result['ensemble'],
                           ms_segm_result['ensemble'],
                           ms_6dof_result['ensemble'])
            elif self.with_mask:
                results = (ms_bbox_result['ensemble'],
                           ms_segm_result['ensemble'])
            else:
                results = ms_bbox_result['ensemble']
        else:
            if self.with_mask:
                results = {
                    stage: (ms_bbox_result[stage], ms_segm_result[stage])
                    for stage in ms_bbox_result
                }
            else:
                results = ms_bbox_result

        return results

    def aug_test(self, imgs, img_metas, proposals=None, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        if self.with_semantic:
            semantic_feats = [
                self.semantic_head(feat)[1]
                for feat in self.extract_feats(imgs)
            ]
        else:
            semantic_feats = [None] * len(img_metas)

        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)

        rcnn_test_cfg = self.test_cfg.rcnn
        aug_bboxes = []
        aug_scores = []
        for x, img_meta, semantic in zip(
                self.extract_feats(imgs), img_metas, semantic_feats):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])
            for i in range(self.num_stages):
                bbox_head = self.bbox_head[i]
                cls_score, bbox_pred = self._bbox_forward_test(
                    i, x, rois, semantic_feat=semantic)
                ms_scores.append(cls_score)

                if i < self.num_stages - 1:
                    bbox_label = cls_score.argmax(dim=1)
                    rois = bbox_head.regress_by_class(rois, bbox_label,
                                                      bbox_pred, img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[]
                               for _ in range(self.mask_head[-1].num_classes -
                                              1)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta, semantic in zip(
                        self.extract_feats(imgs), img_metas, semantic_feats):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip)
                    mask_rois = bbox2roi([_bboxes])
                    mask_feats = self.mask_roi_extractor[-1](
                        x[:len(self.mask_roi_extractor[-1].featmap_strides)],
                        mask_rois)
                    if self.with_semantic:
                        semantic_feat = semantic
                        mask_semantic_feat = self.semantic_roi_extractor(
                            [semantic_feat], mask_rois)
                        if mask_semantic_feat.shape[-2:] != mask_feats.shape[
                                -2:]:
                            mask_semantic_feat = F.adaptive_avg_pool2d(
                                mask_semantic_feat, mask_feats.shape[-2:])
                        mask_feats += mask_semantic_feat
                    last_feat = None
                    for i in range(self.num_stages):
                        mask_head = self.mask_head[i]
                        if self.mask_info_flow:
                            mask_pred, last_feat = mask_head(
                                mask_feats, last_feat)
                        else:
                            mask_pred = mask_head(mask_feats)
                        aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg.rcnn)

                ori_shape = img_metas[0][0]['ori_shape']
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=1.0,
                    rescale=False)
            return bbox_result, segm_result
        else:
            return bbox_result
