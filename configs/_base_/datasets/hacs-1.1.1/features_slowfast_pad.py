dataset_type = "AnetPaddingDataset"
annotation_path = "data/hacs-1.1.1/annotations/HACS_segments_v1.1.1.json"
class_map = "data/hacs-1.1.1/annotations/category_idx.txt"
data_path = "data/hacs-1.1.1/features/slowfast101_15fps_stride8_len32_hacs/"
block_list = data_path + "missing_files.txt"

pad_len = 960

dataset = dict(
    train=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="training",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=True,
        feature_stride=8,
        sample_stride=1,  # 1x8=8
        offset_frames=16,
        fps=15,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="RandomTrunc", trunc_len=pad_len, trunc_thresh=0.5, crop_ratio=[0.9, 1.0]),
            dict(type="Rearrange", keys=["feats"], ops="t c-> c t"),
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
        filter_gt=False,
        feature_stride=8,
        sample_stride=1,  # 1x8=8
        offset_frames=16,
        fps=15,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="Padding", length=pad_len),
            dict(type="Rearrange", keys=["feats"], ops="t c-> c t"),
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
        feature_stride=8,
        sample_stride=1,  # 1x8=8
        offset_frames=16,
        fps=15,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="Padding", length=pad_len),
            dict(type="Rearrange", keys=["feats"], ops="t c-> c t"),
            dict(type="Collect", inputs="feats", keys=["masks"]),
        ],
    ),
)


evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    ground_truth_filename=annotation_path,
)
