_base_ = [
    "../_base_/datasets/thumos-14/features_tsn_sw.py",  # dataset config
    "../_base_/models/gtad.py",  # model config
]

window_size = 256
dataset = dict(
    train=dict(
        feature_stride=1,
        sample_stride=5,
        window_size=window_size,
        window_overlap_ratio=0.5,
        ioa_thresh=0.9,
    ),
    val=dict(
        feature_stride=1,
        sample_stride=5,
        window_size=window_size,
        window_overlap_ratio=0.5,
        ioa_thresh=0.9,
    ),
    test=dict(
        feature_stride=1,
        sample_stride=5,
        window_size=window_size,
        window_overlap_ratio=0.5,
    ),
)

model = dict(
    projection=dict(in_channels=2048),
    roi_head=dict(
        proposal_generator=dict(dscale=64, tscale=256),
        proposal_roi_extractor=dict(dscale=64, tscale=256, roi_size=16, context_size=16),
        proposal_head=dict(kernel_size=3),
    ),
)

solver = dict(
    train=dict(batch_size=8, num_workers=4),
    val=dict(batch_size=8, num_workers=4),
    test=dict(batch_size=8, num_workers=4),
    clip_grad_norm=1,
)

optimizer = dict(type="Adam", lr=1e-4, weight_decay=1e-4)
scheduler = dict(type="MultiStepLR", milestones=[5], gamma=0.1, max_epoch=8)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.3,
        max_seg_num=200,
        min_score=0.0001,
        multiclass=False,
        voting_thresh=0.95,  #  set 0 to disable
    ),
    external_cls=dict(
        type="UntrimmedNetTHUMOSClassifier",
        path="data/thumos-14/classifiers/uNet_test.npy",
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

work_dir = "exps/thumos/gtad_tsn_sw256"
