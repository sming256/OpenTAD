_base_ = [
    "../_base_/datasets/hacs-1.1.1/features_slowfast_resize.py",  # dataset config
    "../_base_/models/bmn.py",  # model config
]

tscale = 192
dataset = dict(
    train=dict(resize_length=tscale),
    val=dict(resize_length=tscale),
    test=dict(resize_length=tscale),
)

model = dict(
    projection=dict(in_channels=2304),
    roi_head=dict(
        proposal_generator=dict(tscale=tscale, dscale=tscale),
        proposal_roi_extractor=dict(tscale=tscale, dscale=tscale),
    ),
)

solver = dict(
    train=dict(batch_size=16, num_workers=4),
    val=dict(batch_size=16, num_workers=4),
    test=dict(batch_size=16, num_workers=4),
    clip_grad_norm=1,
)

optimizer = dict(
    type="Adam",
    lr=4e-4,
    weight_decay=1e-4,
)
scheduler = dict(
    type="MultiStepLR",
    milestones=[5],
    gamma=0.1,
    max_epoch=6,
)


inference = dict(
    test_epoch=5,
    load_from_raw_predictions=False,
    save_raw_prediction=False,
)

post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=100,
        iou_threshold=0,  # does not matter when use soft nms
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
)

work_dir = "exps/hacs/bmn_slowfast_192"
