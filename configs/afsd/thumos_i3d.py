_base_ = [
    "../_base_/datasets/thumos-14/features_i3d_sw.py",
    "../_base_/models/afsd.py",
]

dataset = dict(
    train=dict(sample_stride=3, window_size=64, window_overlap_ratio=0.875),
    val=dict(sample_stride=3, window_size=64, window_overlap_ratio=0.5),
    test=dict(sample_stride=3, window_size=64, window_overlap_ratio=0.5),
)

model = dict(
    neck=dict(in_channels=2048, frame_num=256),
    rpn_head=dict(frame_num=256, feat_t=256 // 4, num_classes=21, fpn_strides=[1, 1, 1, 1, 1, 1]),
    roi_head=dict(num_classes=21, overlap_thresh=0.5, loc_weight=10, loc_bounded=False, use_smooth_l1=False),
)

solver = dict(
    train=dict(batch_size=1, num_workers=4),
    val=dict(batch_size=1, num_workers=4),
    test=dict(batch_size=1, num_workers=4),
    clip_grad_norm=0.1,
)

optimizer = dict(type="AdamW", lr=1e-5, weight_decay=1e-3)
scheduler = dict(type="MultiStepLR", milestones=[16], gamma=0.1, max_epoch=16)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=2000,
        min_score=0.01,
        multiclass=True,
        voting_thresh=0.95,  #  set 0 to disable
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=2000,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
)

work_dir = "exps/thumos/afsd_i3d"
