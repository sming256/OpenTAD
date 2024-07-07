_base_ = [
    "../_base_/datasets/fineaction/features_internvideo_pad.py",  # dataset config
    "../_base_/models/dyfadet.py",  # model config
]

data_path = "data/fineaction/features/fineaction_mae_g/"
block_list = data_path + "missing_files.txt"
dataset = dict(
    train=dict(data_path=data_path, block_list=block_list, class_agnostic=True),
    val=dict(data_path=data_path, block_list=block_list, class_agnostic=True),
    test=dict(data_path=data_path, block_list=block_list, class_agnostic=True),
)

model = dict(
    projection=dict(
        in_channels=1408,
        out_channels=256,
        arch=(2, 2, 6),
        use_abs_pe=True,
        max_seq_len=2304,
        mlp_dim=2048,
        k=1.3,
        init_conv_vars=0.2,
        encoder_win_size=15,
        input_noise=0.5,
    ),
    neck=dict(in_channels=256, out_channels=256, num_levels=7),
    rpn_head=dict(
        in_channels=256,
        feat_channels=512,
        num_classes=1,
        prior_generator=dict(
            strides=[1, 2, 4, 8, 16, 32, 64],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 10000)],
        ),
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
        path="./data/fineaction/classifiers/new_swinB_1x1x256_views2x3_max_label_avg_prob.json",
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

work_dir = "exps/fineaction/dyfadet_videomaev2_g"
