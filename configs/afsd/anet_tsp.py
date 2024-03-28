_base_ = [
    "../_base_/datasets/activitynet-1.3/features_tsp_resize.py",  # dataset config
    "../_base_/models/afsd.py",  # model config
]

dataset = dict(
    train=dict(resize_length=96),
    val=dict(resize_length=96),
    test=dict(resize_length=96),
)

model = dict(neck=dict(in_channels=512))

solver = dict(
    train=dict(batch_size=16, num_workers=4),
    val=dict(batch_size=16, num_workers=4),
    test=dict(batch_size=16, num_workers=4),
    clip_grad_norm=1,
)

optimizer = dict(type="AdamW", lr=1e-3, weight_decay=1e-4)
scheduler = dict(type="LinearWarmupMultiStepLR", warmup_epoch=1, milestones=[5], gamma=0.1, max_epoch=10)

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
    logging_interval=200,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=5,
)

work_dir = "exps/anet/afsd_anet_tsp_96"
