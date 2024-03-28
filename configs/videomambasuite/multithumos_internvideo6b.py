_base_ = [
    "../_base_/datasets/multithumos/features_i3d_pad.py",  # dataset config
    "../_base_/models/videomambasuite.py",  # model config
]

trunc_len = 2304
data_path = "data/thumos-14/features/thumos14_6b/"
dataset = dict(
    train=dict(
        data_path=data_path,
        block_list=None,
        offset_frames=8,
        pipeline=[
            dict(type="LoadFeats", feat_format="pt", suffix="_spatial_feature"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(
                type="RandomTrunc",
                trunc_len=trunc_len,
                trunc_thresh=0.5,
                crop_ratio=[0.9, 1.0],
            ),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(
                type="Collect",
                inputs="feats",
                keys=["masks", "gt_segments", "gt_labels"],
            ),
        ],
    ),
    val=dict(
        data_path=data_path,
        block_list=None,
        offset_frames=8,
        pipeline=[
            dict(type="LoadFeats", feat_format="pt", suffix="_spatial_feature"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(
                type="Collect",
                inputs="feats",
                keys=["masks", "gt_segments", "gt_labels"],
            ),
        ],
    ),
    test=dict(
        data_path=data_path,
        block_list=None,
        offset_frames=8,
        pipeline=[
            dict(type="LoadFeats", feat_format="pt", suffix="_spatial_feature"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks"]),
        ],
    ),
)

model = dict(
    projection=dict(in_channels=3200, arch=(3, 0, 5), input_pdrop=0.1),
    rpn_head=dict(num_classes=65),
)

solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=1, num_workers=1),
    test=dict(batch_size=1, num_workers=1),
    clip_grad_norm=1,
    ema=True,
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=65)

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
    logging_interval=20,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=39,
    end_epoch=55,
)

work_dir = "exps/multithumos/videomambasuite_internvideo6b"
