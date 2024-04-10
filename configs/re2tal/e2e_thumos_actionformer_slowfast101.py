_base_ = [
    "../_base_/datasets/thumos-14/e2e_train_trunc_test_sw_256x224x224.py",  # dataset config
    "../_base_/models/actionformer.py",  # model config
]

window_size = 512
dataset = dict(
    train=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=window_size,
                trunc_thresh=0.75,
                crop_ratio=[0.9, 1.0],
                scale_factor=2,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 256)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(224, 224), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        window_size=window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=2),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 224)),
            dict(type="mmaction.CenterCrop", crop_size=224),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        window_size=window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=2),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 224)),
            dict(type="mmaction.CenterCrop", crop_size=224),
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
            type="ResNet3dSlowFast",
            pretrained=None,
            resample_rate=4,  # tau
            speed_ratio=4,  # alpha
            channel_ratio=8,  # beta_inv
            slow_pathway=dict(
                type="resnet3d",
                depth=101,
                pretrained=None,
                lateral=True,
                fusion_kernel=7,
                conv1_kernel=(1, 7, 7),
                dilations=(1, 1, 1, 1),
                conv1_stride_t=1,
                pool1_stride_t=1,
                inflate=(0, 0, 1, 1),
                norm_eval=True,
                with_cp=True,
            ),
            fast_pathway=dict(
                type="resnet3d",
                depth=101,
                pretrained=None,
                lateral=False,
                base_channels=8,
                conv1_kernel=(5, 7, 7),
                conv1_stride_t=1,
                pool1_stride_t=1,
                norm_eval=True,
                with_cp=True,
            ),
        ),
        data_preprocessor=dict(
            type="mmaction.ActionDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape="NCTHW",
        ),
        custom=dict(
            pretrain="https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb_20220818-9c0e09bd.pth",
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b t c tt h w -> b c tt", reduction="mean"),
                dict(type="Interpolate", keys=["feats"], size=window_size),
            ],
            norm_eval=True,  # set all norm layers to eval mode, default is True
            freeze_backbone=False,  # whether to freeze the backbone, default is False
        ),
    ),
    projection=dict(in_channels=2304, max_seq_len=window_size, attn_cfg=dict(n_mha_win_size=-1)),
)

solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=2, num_workers=2),
    test=dict(batch_size=2, num_workers=2),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=True,
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(lr=1e-5, weight_decay=0.05),
)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=1500)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.7,
        max_seg_num=2000,
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.7,  #  set 0 to disable
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=5,
    val_start_epoch=79,
    end_epoch=120,
)

work_dir = "exps/thumos/e2e_actionformer_slowfast101_frame1024"
