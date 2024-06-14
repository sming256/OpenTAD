annotation_path = "data/charades/annotations/charades.json"
class_map = "data/charades/annotations/category_idx.txt"
data_path = "data/charades/features/videomae-large_16x4x1_img224_stride4_len16_interval1_charades_30fps/"
block_list = None

window_size = 512

dataset = dict(
    train=dict(
        type="EpicKitchensPaddingDataset",
        ann_file=annotation_path,
        subset_name="training",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # dataloader setting
        fps=30,
        feature_stride=4,
        sample_stride=1,
        offset_frames=2,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="RandomTrunc", trunc_len=window_size, trunc_thresh=0.5, crop_ratio=[0.9, 1.0]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        type="EpicKitchensSlidingDataset",
        ann_file=annotation_path,
        subset_name="testing",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # dataloader setting
        window_size=window_size,
        fps=30,
        feature_stride=4,
        sample_stride=1,
        offset_frames=2,
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
        type="EpicKitchensSlidingDataset",
        ann_file=annotation_path,
        subset_name="testing",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        # dataloader setting
        window_size=window_size,
        fps=30,
        feature_stride=4,
        sample_stride=1,
        offset_frames=2,
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
    subset="testing",
    tiou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ground_truth_filename=annotation_path,
)
