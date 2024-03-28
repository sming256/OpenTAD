_base_ = [
    "../_base_/datasets/activitynet-1.3/e2e_resize_768_1x224x224.py",  # dataset config
    "../_base_/models/afsd.py",  # model config
]

resize_length = 96
dataset = dict(
    train=dict(
        resize_length=resize_length,
        block_list=None,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4", prefix="v_"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="resize", scale_factor=8),  # load 96x8=768 frames
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 109)),
            dict(type="mmaction.RandomCrop", size=96),
            dict(type="mmaction.Flip", flip_ratio=0.5, direction="horizontal"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        resize_length=resize_length,
        block_list=None,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4", prefix="v_"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="resize", scale_factor=8),  # load 96x8=768 frames
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 96)),
            dict(type="mmaction.CenterCrop", crop_size=96),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        resize_length=resize_length,
        block_list=None,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4", prefix="v_"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="resize", scale_factor=8),  # load 96x8=768 frames
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
            ],
            # the post_processing pipeline will reduce the feature to [bs,2048,96,3,3]
            norm_eval=True,  # set all norm layers to eval mode, default is True
            freeze_backbone=False,  # whether to freeze the backbone, default is False
        ),
    ),
    neck=dict(type="AFSDNeck", in_channels=2048, e2e=True),
)

solver = dict(
    train=dict(batch_size=16, num_workers=8),
    val=dict(batch_size=16, num_workers=8),
    test=dict(batch_size=16, num_workers=8),
    clip_grad_norm=1,
)

optimizer = dict(
    type="AdamW",
    lr=5e-4,
    weight_decay=1e-4,
    backbone=dict(lr=1e-5, weight_decay=1e-4),
)
scheduler = dict(type="LinearWarmupMultiStepLR", warmup_epoch=1, milestones=[5], gamma=0.1, max_epoch=15)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.85,
        max_seg_num=100,
        min_score=0.001,
        multiclass=False,
        voting_thresh=0.9,  #  set 0 to disable
    ),
    external_cls=dict(
        type="CUHKANETClassifier",
        path="data/activitynet-1.3/classifiers/cuhk_val_simp_7.json",
        topk=2,
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=100,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=5,
)

work_dir = "exps/anet/afsd_anet_i3d_768x96x96_bs16"
