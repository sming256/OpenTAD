_base_ = [
    "../_base_/datasets/charades/features_i3d_pad.py",  # dataset config
    "../_base_/models/actionformer.py",  # model config
]

model = dict(
    projection=dict(in_channels=1024),
    rpn_head=dict(num_classes=157),
)

solver = dict(
    train=dict(batch_size=16, num_workers=4),
    val=dict(batch_size=16, num_workers=4),
    test=dict(batch_size=16, num_workers=4),
    clip_grad_norm=1,
    ema=True,
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=15)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    pre_nms_topk=8000,
    pre_nms_thresh=0.001,
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=8000,
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.7,  #  set 0 to disable
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=200,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=5,
    end_epoch=10,
)

work_dir = "exps/charades/actionformer_i3d_rgb"
