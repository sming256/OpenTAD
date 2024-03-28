dataset_type = "AnetPaddingDataset"
annotation_path = "data/activitynet-1.3/annotations/activity_net.v1-3.min.json"
class_map = "data/activitynet-1.3/annotations/category_idx.txt"
data_path = "data/activitynet-1.3/features/anet_tsp_npy_unresize/"
block_list = "data/activitynet-1.3/features/anet_tsp_npy_unresize/missing_files.txt"

pad_len = 768

dataset = dict(
    train=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="training",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=True,
        feature_stride=16,
        sample_stride=1,  # 1x16=16
        offset_frames=8,
        fps=15,
        class_agnostic=True,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy", prefix="v_"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="RandomTrunc", trunc_len=pad_len, trunc_thresh=0.5, crop_ratio=[0.9, 1.0]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="validation",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=True,
        feature_stride=16,
        sample_stride=1,  # 1x16=16
        offset_frames=8,
        fps=15,
        class_agnostic=True,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy", prefix="v_"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="Padding", length=pad_len),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="validation",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        feature_stride=16,
        sample_stride=1,  # 1x16=16
        offset_frames=8,
        fps=15,
        class_agnostic=True,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy", prefix="v_"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="Padding", length=pad_len),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks"]),
        ],
    ),
)

evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    ground_truth_filename=annotation_path,
    blocked_videos="data/activitynet-1.3/annotations/blocked.json",
)
