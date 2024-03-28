_base_ = [
    "../_base_/datasets/activitynet-1.3/features_tsp_resize.py",  # dataset config
    "../_base_/models/vsgn.py",  # model config
]


tscale = 256

dataset = dict(
    train=dict(resize_length=tscale),
    val=dict(resize_length=tscale),
    test=dict(resize_length=tscale),
)

model = dict(
    projection=dict(in_channels=512),
    rpn_head=dict(
        anchor_generator=dict(tscale=tscale, anchor_scale=[3, 7.5]),
        loss_cls=dict(type="FocalLoss", gamma=2.0, alpha=0.35),
    ),
    roi_head=dict(roi_extractor=dict(tscale=tscale)),
)

solver = dict(
    train=dict(batch_size=32, num_workers=4),
    val=dict(batch_size=32, num_workers=4),
    test=dict(batch_size=32, num_workers=4),
    clip_grad_norm=1,
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=1e-4)
scheduler = dict(type="MultiStepLR", milestones=[15], gamma=0.1, max_epoch=15)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=100,
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
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=7,
)
work_dir = "exps/anet/vsgn_tsp_256"
