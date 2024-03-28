_base_ = [
    "../_base_/datasets/activitynet-1.3/e2e_resize_768_1x224x224.py",  # dataset config
    "../_base_/models/actionformer.py",  # model config
]

resize_length = 192
dataset = dict(
    train=dict(
        resize_length=resize_length,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4", prefix="v_"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="resize", scale_factor=4),
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
        resize_length=resize_length,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4", prefix="v_"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="resize", scale_factor=4),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 224)),
            dict(type="mmaction.CenterCrop", crop_size=224),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        resize_length=resize_length,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4", prefix="v_"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="resize", scale_factor=4),
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
            type="SwinTransformer3D_inv",
            patch_size=(2, 4, 4),
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=(8, 7, 7),
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            patch_norm=True,
            inv_mode=["custom_backprop", "custom_backprop", "custom_backprop", "custom_backprop"],
        ),
        data_preprocessor=dict(
            type="mmaction.ActionDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape="NCTHW",
        ),
        custom=dict(
            pretrain="pretrained/invswin-tiny-p244-w877_kinetics400_pre_k400_ft_e30.pth",
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b t c tt h w -> b c tt", reduction="mean"),
                dict(type="Interpolate", keys=["feats"], size=resize_length),
            ],
            norm_eval=True,  # set all norm layers to eval mode, default is True
            freeze_backbone=False,  # whether to freeze the backbone, default is False
        ),
    ),
    projection=dict(
        in_channels=768,
        out_channels=256,
        attn_cfg=dict(n_mha_win_size=-1),
        use_abs_pe=True,
        max_seq_len=resize_length,
    ),
    neck=dict(in_channels=256, out_channels=256),
    rpn_head=dict(
        in_channels=256,
        feat_channels=256,
        num_classes=1,
        label_smoothing=0.1,
        loss_weight=2.0,
        loss_normalizer=200,
    ),
)

solver = dict(
    train=dict(batch_size=8, num_workers=4),
    val=dict(batch_size=8, num_workers=4),
    test=dict(batch_size=8, num_workers=4),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=True,
)

optimizer = dict(
    type="AdamW",
    lr=1e-3,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(lr=1e-5, weight_decay=0.05),
)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=15)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.7,
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
    val_loss_interval=-1,  # do not validate during end-to-end training, waste time
    val_eval_interval=1,
    val_start_epoch=8,
)

work_dir = "exps/anet/re2tal_e2e_actionformer_swin_tiny_frame768"
