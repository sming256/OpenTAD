_base_ = [
    "../_base_/datasets/thumos-14/features_tsn_sw.py",  # dataset config
    "../_base_/models/vsgn.py",  # model config
]

window_size = 1280
dataset = dict(
    train=dict(window_overlap_ratio=0.75, window_size=window_size, sample_stride=1),
    val=dict(window_overlap_ratio=0.75, window_size=window_size, sample_stride=1),
    test=dict(window_overlap_ratio=0.75, window_size=window_size, sample_stride=1),
)

pyramid_levels = [4, 8, 16, 32, 64]
model = dict(
    projection=dict(in_channels=2048, pyramid_levels=pyramid_levels),
    rpn_head=dict(
        anchor_generator=dict(tscale=window_size, pyramid_levels=pyramid_levels, anchor_scale=[1, 1.5]),
        loss_cls=dict(type="BalancedCELoss"),
        num_classes=21,
        iou_thr=0.5,
    ),
    roi_head=dict(roi_extractor=dict(tscale=window_size, base_stride=4)),
)


solver = dict(
    train=dict(batch_size=32, num_workers=4),
    val=dict(batch_size=32, num_workers=4),
    test=dict(batch_size=32, num_workers=4),
    clip_grad_norm=1,
)

optimizer = dict(type="Adam", lr=5e-5, weight_decay=1e-4)
scheduler = dict(type="MultiStepLR", milestones=[15], max_epoch=10)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.4,
        multiclass=True,
        max_seg_num=200,
        min_score=0.0001,
        voting_thresh=0.95,  #  set 0 to disable
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=5,
)

work_dir = "exps/thumos/vsgn_tsn_sw1280"
