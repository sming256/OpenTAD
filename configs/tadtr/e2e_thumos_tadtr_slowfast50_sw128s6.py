_base_ = [
    "../_base_/datasets/thumos-14/e2e_sw_256x224x224.py",  # dataset config
    "../_base_/models/tadtr.py",  # model config
]

window_size = 128
dataset = dict(
    train=dict(
        feature_stride=6,
        window_overlap_ratio=0.25,
        window_size=window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=2),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 110)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(96, 96), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        feature_stride=6,
        window_overlap_ratio=0.25,
        window_size=window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=2),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 96)),
            dict(type="mmaction.CenterCrop", crop_size=96),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        feature_stride=6,
        window_overlap_ratio=0.75,
        window_size=window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=2),
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
            type="mmaction.ResNet3dSlowFast",
            pretrained=None,
            resample_rate=8,  # tau
            speed_ratio=8,  # alpha
            channel_ratio=8,  # beta_inv
            slow_pathway=dict(
                type="resnet3d",
                depth=50,
                pretrained=None,
                lateral=True,
                conv1_kernel=(1, 7, 7),
                dilations=(1, 1, 1, 1),
                conv1_stride_t=1,
                pool1_stride_t=1,
                inflate=(0, 0, 1, 1),
                norm_eval=True,
            ),
            fast_pathway=dict(
                type="resnet3d",
                depth=50,
                pretrained=None,
                lateral=False,
                base_channels=8,
                conv1_kernel=(5, 7, 7),
                conv1_stride_t=1,
                pool1_stride_t=1,
                norm_eval=True,
            ),
        ),
        data_preprocessor=dict(
            type="mmaction.ActionDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape="NCTHW",
        ),
        custom=dict(
            pretrain="https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth",
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b t c tt h w -> b c tt", reduction="mean"),
                dict(type="Interpolate", keys=["feats"], size=window_size),
            ],
            norm_eval=True,  # set all norm layers to eval mode, default is True
            freeze_backbone=False,  # whether to freeze the backbone, default is False
        ),
    ),
    projection=dict(in_channels=2304),
)

solver = dict(
    train=dict(batch_size=4, num_workers=4),
    val=dict(batch_size=4, num_workers=4),
    test=dict(batch_size=4, num_workers=0),
    clip_grad_norm=0.1,
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=1e-4,
    paramwise=True,
    backbone=dict(lr=5e-6, weight_decay=1e-4),
)
scheduler = dict(type="MultiStepLR", milestones=[14], gamma=0.1, max_epoch=17)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.4,
        max_seg_num=2000,
        multiclass=True,
        voting_thresh=0.95,  #  set 0 to disable
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=100,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=8,
)

work_dir = "exps/thumos/tadtr_e2e_slowfast50_sw128s6"
