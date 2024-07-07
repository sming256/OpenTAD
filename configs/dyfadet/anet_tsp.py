_base_ = [
    "../_base_/datasets/activitynet-1.3/features_tsp_pad.py",  # dataset config
    "../_base_/models/dyfadet.py",  # model config
]

model = dict(
    projection=dict(
        in_channels=512,
        out_channels=256,
        use_abs_pe=True,
        max_seq_len=768,
        mlp_dim=2048,
        k=1.2,
        init_conv_vars=0.15,
        encoder_win_size=15,
        input_noise=0.7,
    ),
    neck=dict(in_channels=256, out_channels=256),
    rpn_head=dict(
        in_channels=256,
        feat_channels=512,
        num_classes=1,
        label_smoothing=0.1,
        loss_normalizer=400,
    ),
)

solver = dict(
    train=dict(batch_size=8, num_workers=4),
    val=dict(batch_size=8, num_workers=4),
    test=dict(batch_size=8, num_workers=4),
    clip_grad_norm=0.4,
    ema=True,
)

optimizer = dict(type="AdamW", lr=5e-4, weight_decay=0.025, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=10, max_epoch=16, eta_min=5e-5)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.7,  # test 0.75
        max_seg_num=200,
        min_score=0.001,
        multiclass=False,
        voting_thresh=0.95,  #  set 0 to disable
    ),
    external_cls=dict(
        type="StandardClassifier",
        path="data/activitynet-1.3/classifiers/new_3ensemble_uniformerv2_large_only_global_anet_16x10x3.json",
        topk=2,
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

work_dir = "exps/anet/dyfadet_tsp"
