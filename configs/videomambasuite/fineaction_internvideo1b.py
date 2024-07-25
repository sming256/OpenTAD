_base_ = [
    "../_base_/datasets/fineaction/features_internvideo_pad.py",  # dataset config
    "../_base_/models/videomambasuite.py",  # model config
]

trunc_len = 2304
data_path = "data/fineaction/features/fineaction_1b/"
dataset = dict(
    train=dict(
        data_path=data_path,
        block_list=["v_00004011", "v_00006080", "v_00002783"],  # dirty annotation
        feature_stride=4,
        sample_stride=2,
        offset_frames=8,
        fps=30,
        pipeline=[
            dict(type="LoadFeats", feat_format="pt", suffix="_spatial_pool_feature_5"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="RandomTrunc", trunc_len=trunc_len, trunc_thresh=0.5, crop_ratio=[0.9, 1.0]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        data_path=data_path,
        block_list=None,
        feature_stride=4,
        sample_stride=2,
        offset_frames=8,
        fps=30,
        pipeline=[
            dict(type="LoadFeats", feat_format="pt", suffix="_spatial_pool_feature_5"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        data_path=data_path,
        block_list=None,
        feature_stride=4,
        sample_stride=2,
        offset_frames=8,
        fps=30,
        pipeline=[
            dict(type="LoadFeats", feat_format="pt", suffix="_spatial_pool_feature_5"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks"]),
        ],
    ),
)
model = dict(
    projection=dict(
        in_channels=1408,
        out_channels=512,
        arch=(2, 2, 7),
        max_seq_len=2304,
        input_pdrop=0.1,
    ),
    neck=dict(in_channels=512, out_channels=512, num_levels=8),
    rpn_head=dict(
        in_channels=512,
        feat_channels=512,
        num_classes=106,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1, 2, 4, 8, 16, 32, 64, 128],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 10000)],
        ),
    ),
)

solver = dict(
    train=dict(batch_size=16, num_workers=4),
    val=dict(batch_size=1, num_workers=0),
    test=dict(batch_size=1, num_workers=0),
    clip_grad_norm=1.0,
    ema=True,
)

optimizer = dict(type="AdamW", lr=1e-3, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=25)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.75,
        max_seg_num=200,
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
    val_start_epoch=10,
)

work_dir = "exps/fineaction/videomambasuite_internvideo1b"
