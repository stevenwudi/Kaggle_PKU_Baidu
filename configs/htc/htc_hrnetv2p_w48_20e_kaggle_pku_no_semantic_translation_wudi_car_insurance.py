# model settings
model = dict(
    type='HybridTaskCascade',
    num_stages=3,
    interleaved=True,
    mask_info_flow=True,
    car_cls_info_flow=False,
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384)))),
    neck=dict(
        type='HRFPN',
        in_channels=[48, 96, 192, 384],
        out_channels=256),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=[
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=81,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2],
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=81,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.05, 0.05, 0.1, 0.1],
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=81,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.033, 0.033, 0.067, 0.067],
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
    ],
    mask_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    mask_head=dict(
        type='HTCMaskHead',
        num_convs=4,
        in_channels=256,
        conv_out_channels=256,
        num_classes=81,
        loss_mask=dict(
            type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),

    with_semantic_loss=False,
    with_car_cls_rot=True,
    with_translation=True,
    # This is DI WU's customised model
    semantic_fusion=('bbox', 'mask', 'car_cls_rot'),
    car_cls_rot_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),

    car_cls_rot_head=dict(
        type='SharedCarClsRotHead',
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=14,
        num_classes=34,  # There are total 34 car classes
        reg_class_agnostic=True,
        loss_car_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_quaternion=dict(type='L1', beta=1.0, loss_weight=1.0)),

    translation_head=dict(
        type='SharedTranslationHead',
        in_channels_bboxes=4,
        in_channels_carclsrot=1024,
        fc_out_channels=100,
        num_translation_reg=3,
        bbox_relative=False,  # if bbox_relative=False, then it requires training/test input the same
        translation_bboxes_regression=False,  # If set to True, we will have a SSD like offset regression
        bboxes_regression=dict(type='maxIoU', iou_thresh=0.1),
        #bboxes_regression=dict(type='allIoU', iou_thresh=0.1),  # This is only effective during test
        loss_translation=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),

    bayesian_weight_learning=True,  # If set to true, the loss weight coefficient will be updated.

)

# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=[
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.7,
                min_pos_iou=0.7,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)
    ],
    stage_loss_weights=[1, 0.5, 0.25],

    car_cls_weight=1.0,
    rot_weight=10.,
    translation_weight=10.,
)
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.001,
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=100,
        mask_thr_binary=0.5),
    keep_all_stages=False,
)
# dataset settings
dataset_type = 'KagglePKUDataset'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# Add albumentation transform
# albu_train_transforms = [
# #     dict(type='RandomBrightnessContrast', brightness_limit=0.2, contrast_limit=0.5, p=0.2),
# #     dict(type='GaussianBlur', blur_limit=20, p=0.1),
# #     dict(type='GaussNoise', var_limit=(10, 80.), p=0.1),
# #     dict(
# #         type='OneOf',
# #         transforms=[
# #             dict(
# #                 type='RGBShift',
# #                 r_shift_limit=30,
# #                 g_shift_limit=30,
# #                 b_shift_limit=30,
# #                 p=0.2),
# #             dict(
# #                 type='HueSaturationValue',
# #                 hue_shift_limit=20,
# #                 sat_shift_limit=20,
# #                 val_shift_limit=20,
# #                 p=0.1)
# #         ],
# #         p=0.1),
# ]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True,
         with_carcls_rot=True, with_translation=True, with_camera_rot=True),
    dict(type='CameraRotation'),
    dict(type='CropBottom', bottom_half=1480),
    dict(type='Resize', img_scale=(1664, 576), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    # dict(
    #     type='Albu',
    #     transforms=albu_train_transforms,
    #     update_pad_shape=False,
    #     skip_img_without_anno=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks',
               'carlabels', 'quaternion_semispheres', 'translations',
               'scale_factor']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropBottom', bottom_half=1480),
    #dict(type='CropCentreResize', top=50, bottom=100, left=25, right=50),
    #dict(type='CropCentreResize', top=100, bottom=250, left=50, right=100),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 480),  # (576, 1600, 3)
        flip=False,  # test pipelines doest not need this
        transforms=[
            dict(type='Resize', img_scale=(640, 480), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.),   # We always want to have this flip_ratio=1.0 for test
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# data_root = '/data/Kaggle/pku-autonomous-driving/'
data_root = '/data/Kaggle/ApolloScape_3D_car/train/'
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        #ann_file='/data/cyh/kaggle/kaggle_apollo_combine_6692.json',
        # ann_file=data_root + 'apollo_kaggle_combined_6725_wudi.json',
        ann_file='/data/Kaggle/kaggle_apollo_combined_6691_origin.json',  # 6691 means the final cleaned data
        img_prefix=data_root + 'train_images/',
        pipeline=train_pipeline,
        rotation_augmenation=True),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/data/Kaggle/pku-autonomous-driving/validation.csv',
        img_prefix='/data/Kaggle/pku-autonomous-driving/validation_images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="",
        #ann_file='/data/Kaggle/ApolloScape_3D_car/train/split/validation-list.txt',
        #img_prefix='/data/Kaggle/pku-autonomous-driving/validation_images_RandomBrightnessContrast',  # valid variation
        #img_prefix='/data/Kaggle/pku-autonomous-driving/validation_images_RGBShift',  # valid variation
        #img_prefix='/data/Kaggle/pku-autonomous-driving/validation_images_JpegCompression',  # valid variation
        #img_prefix='/data/Kaggle/pku-autonomous-driving/validation_images_GaussianBlur',  # valid variation
        #img_prefix='/data/Kaggle/pku-autonomous-driving/validation_images_GaussNoise',  # valid variation
        #img_prefix='/data/Kaggle/pku-autonomous-driving/validation_images_RandomContrast',  # valid variation
        #img_prefix='/data/Kaggle/pku-autonomous-driving/validation_images_HueSaturationValue',  # valid variation
        #img_prefix='/data/Kaggle/pku-autonomous-driving/validation_images_CLAHE',  # valid variation
        #img_prefix='/data/Kaggle/pku-autonomous-driving/validation_images',  # We create 400 validation images

        #img_prefix='/data/Kaggle/pku-autonomous-driving/test_images',
        #img_prefix='/data/Kaggle/ApolloScape_3D_car/train/images',
        img_prefix='/data/Kaggle/ApolloScape_3D_car/3d-car-understanding-test/test/images',
        pipeline=test_pipeline))


evaluation = dict(
    conf_thresh=0.1,
    interval=1,
)
# optimizer
optimizer = dict(type='Adam', lr=0.0003)  # We increase the learning rate to 3e-4 (It is supposed to be the best practice)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[80, 180])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 200
#dist_params = dict(backend='nccl')
dist_params = dict(backend='nccl', init_method="tcp://127.0.0.1:8001")

log_level = 'INFO'
work_dir = '/data/Kaggle/wudi_data/'
load_from = None
#load_from = '/data/Kaggle/mmdet_pretrained_weights/trimmed_htc_hrnetv2p_w48_20e_kaggle_pku.pth'
#load_from = '/data/Kaggle/wudi_data/Jan07-20-00-59/epoch_5.pth'
#load_from = '/data/Kaggle/checkpoints/all_cwxe99_3070100flip05resumme93Dec29-16-28-48_trimmed_translation.pth'
#load_from = '/data/Kaggle/wudi_data/Jan18-19-45/epoch_116.pth'
resume_from = '/data/Kaggle/checkpoints/all_cwxe99_3070100flip05resumme93Dec29-16-28-48/epoch_100.pth'
#load_from = '/data/Kaggle/wudi_data/Jan29-00-02/epoch_261.pth'
#resume_from = None
workflow = [('train', 1)]

# postprocessing flags here
pkl_postprocessing_restore_xyz = True  # Use YYJ post processing
write_submission = True
valid_eval = False    # evaluate validation set at the end
