_base_ = [
    "../_base_/datasets/hacs-1.1.1/features_slowfast_pad.py",  # dataset config
    "../_base_/models/dyfadet.py",  # model config
]

model = dict(
    projection=dict(
        in_channels=2304,
        out_channels=1024,
        max_seq_len=960,
        use_abs_pe=True,
        mlp_dim=1024,
        k=1.2,
        init_conv_vars=0.1,
        encoder_win_size=3,
        input_noise=0.2,
    ),
    neck=dict(in_channels=1024, out_channels=1024),
    rpn_head=dict(
        in_channels=1024,
        feat_channels=1024,
        num_classes=200,
        head_kernel_size=5,
        label_smoothing=0.1,
        loss_normalizer=400,
    ),
)
solver = dict(
    train=dict(batch_size=8, num_workers=4),
    val=dict(batch_size=8, num_workers=4),
    test=dict(batch_size=8, num_workers=4),
    clip_grad_norm=0.5,
    ema=True,
)

optimizer = dict(type="AdamW", lr=5e-4, weight_decay=0.025, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=7, max_epoch=14, eta_min=5e-4)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.75,
        max_seg_num=250,
        min_score=0.001,
        multiclass=False,
        voting_thresh=0.95,  #  set 0 to disable
    ),
    external_cls=dict(
        type="TCANetHACSClassifier",
        path="data/hacs-1.1.1/classifiers/validation94.32.json",
        topk=3,
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=200,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=10,
)

work_dir = "exps/hacs/dyfadet_slowfast"
