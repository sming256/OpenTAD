_base_ = [
    "../_base_/datasets/thumos-14/e2e_sw_256x224x224.py",
    "../_base_/models/afsd.py",
]

window_size = 64
dataset = dict(
    train=dict(
        feature_stride=12,
        sample_stride=1,
        window_size=window_size,
        window_overlap_ratio=0.875,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=4),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 109)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(96, 96), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        feature_stride=12,
        sample_stride=1,
        window_size=window_size,
        window_overlap_ratio=0.5,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=4),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 96)),
            dict(type="mmaction.CenterCrop", crop_size=96),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        feature_stride=12,
        sample_stride=1,
        window_size=window_size,
        window_overlap_ratio=0.5,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=4),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 96)),
            dict(type="mmaction.CenterCrop", crop_size=96),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
)

model = dict(
    backbone=dict(
        type="mmaction.Recognizer3D",
        backbone=dict(
            type="mmaction.ResNet3d",
            pretrained2d=True,
            pretrained="torchvision://resnet50",
            depth=50,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=2,
            pool1_stride_t=2,
            conv_cfg=dict(type="Conv3d"),
            norm_eval=False,
            inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
            zero_init_residual=False,
        ),
        data_preprocessor=dict(
            type="mmaction.ActionDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape="NCTHW",
        ),
        custom=dict(
            pretrain="https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-dense-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50_8xb8-dense-32x2x1-100e_kinetics400-rgb_20220812-9f46003f.pth",
            # input video shape is [bs,1,3,768,96,96], after the backbone, the feature shape is [bs,1,2048,96,3,3]
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t h w", reduction="mean"),
                dict(type="Interpolate", keys=["feats"], size=(window_size, 3, 3), mode="trilinear"),
            ],
            # the post_processing pipeline will reduce the feature to [bs,2048,96,3,3]
            norm_eval=True,  # set all norm layers to eval mode, default is True
            freeze_backbone=False,  # whether to freeze the backbone, default is False
        ),
    ),
    neck=dict(type="AFSDNeck", in_channels=2048, e2e=True, frame_num=256),
    rpn_head=dict(frame_num=256, feat_t=256 // 4, num_classes=21, fpn_strides=[1, 1, 1, 1, 1, 1]),
    roi_head=dict(num_classes=21, overlap_thresh=0.5, loc_weight=10, loc_bounded=False, use_smooth_l1=False),
)

solver = dict(
    train=dict(batch_size=1, num_workers=1),
    val=dict(batch_size=1, num_workers=1),
    test=dict(batch_size=1, num_workers=1),
    clip_grad_norm=0.1,
)

optimizer = dict(
    type="AdamW",
    lr=1e-5,
    weight_decay=1e-3,
    backbone=dict(lr=1e-5, weight_decay=1e-3),
)
scheduler = dict(type="MultiStepLR", milestones=[16], gamma=0.1, max_epoch=16)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=2000,
        min_score=0.01,
        multiclass=True,
        voting_thresh=0.95,  #  set 0 to disable
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=2000,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
)

work_dir = "exps/thumos/afsd_i3d_256x96x96"
