_base_ = [
    "../_base_/datasets/activitynet-1.3/features_tsp_resize.py",  # dataset config
    "../_base_/models/etad.py",  # model config
]

model = dict(projection=dict(in_channels=512))

solver = dict(
    train=dict(batch_size=16, num_workers=4),
    val=dict(batch_size=16, num_workers=4),
    test=dict(batch_size=16, num_workers=4),
    clip_grad_norm=1,
)

optimizer = dict(type="AdamW", lr=1e-3, weight_decay=1e-4)
scheduler = dict(type="MultiStepLR", milestones=[5], gamma=0.1, max_epoch=6)

inference = dict(test_epoch=5, load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.35,
        max_seg_num=100,
        min_score=0.0001,
        multiclass=False,
        voting_thresh=0,  #  set 0 to disable
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

work_dir = "exps/anet/etad_tsp_128"
