_base_ = [
    "../_base_/datasets/thumos-14/features_i3d_sw.py",
    "../_base_/models/tadtr.py",
]

dataset = dict(
    train=dict(window_size=256, window_overlap_ratio=0.25),
    val=dict(window_size=256, window_overlap_ratio=0.25),
    test=dict(window_size=256, window_overlap_ratio=0.75),
)

solver = dict(
    train=dict(batch_size=8, num_workers=4),
    val=dict(batch_size=8, num_workers=4),
    test=dict(batch_size=8, num_workers=4),
    clip_grad_norm=0.1,
)

optimizer = dict(type="AdamW", lr=2e-4, weight_decay=1e-4, paramwise=True)
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
    logging_interval=300,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=12,
)

work_dir = "exps/thumos/tadtr_i3d"
