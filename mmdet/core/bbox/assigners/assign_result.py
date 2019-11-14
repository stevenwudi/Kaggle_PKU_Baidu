import torch


class AssignResult(object):

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None,
                 assigned_carlabels=None, assigned_quaternion_semispheres=None,
                 assigned_translations=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        self.assigned_carlabels = assigned_carlabels
        self.assigned_quaternion_semispheres = assigned_quaternion_semispheres
        self.assigned_translations = assigned_translations

    def add_gt_(self, gt_labels, carlabels=None, quaternion_semispheres=None, translations=None):
        self_inds = torch.arange(1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])
        self.max_overlaps = torch.cat([self.max_overlaps.new_ones(self.num_gts), self.max_overlaps])
        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])
        if self.assigned_carlabels is not None:
            self.assigned_carlabels = torch.cat([carlabels, self.assigned_carlabels])
        if self.assigned_quaternion_semispheres is not None:
            self.assigned_quaternion_semispheres = torch.cat([quaternion_semispheres, self.assigned_quaternion_semispheres])
        if self.assigned_translations is not None:
            self.assigned_translations = torch.cat([translations, self.assigned_translations])
