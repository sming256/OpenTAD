annotation_path = "data/thumos-14/annotations/thumos_14_anno.json"
class_map = "data/thumos-14/annotations/category_idx.txt"
data_path = "data/thumos-14/features/i3d_actionformer_stride4_thumos/"
block_list = data_path + "missing_files.txt"

window_size = 2304

dataset = dict(
    train=dict(
        type="ThumosPaddingDataset",
        ann_file=annotation_path,
        subset_name="training",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # thumos dataloader setting
        feature_stride=4,
        sample_stride=1,  # 1x4=4
        offset_frames=8,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="RandomTrunc", trunc_len=window_size, trunc_thresh=0.75, crop_ratio=[0.9, 1.0]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="validation",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # thumos dataloader setting
        feature_stride=4,
        sample_stride=1,  # 1x4=4
        window_size=window_size,
        offset_frames=8,
        window_overlap_ratio=0.25,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="validation",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        # thumos dataloader setting
        feature_stride=4,
        sample_stride=1,  # 1x4=4
        window_size=window_size,
        offset_frames=8,
        window_overlap_ratio=0.5,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks"]),
        ],
    ),
)


evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename=annotation_path,
)
