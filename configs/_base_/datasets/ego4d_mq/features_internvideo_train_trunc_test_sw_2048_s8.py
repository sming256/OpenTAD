annotation_path = "data/ego4d/annotations/ego4d_v2_220429.json"
class_map = "data/ego4d/annotations/category_idx.txt"
data_path = "data/ego4d/features/videomae_large_internvideo_img256_stride8_len16_interval1_ego4d/"

window_size = 2048
dataset = dict(
    train=dict(
        type="Ego4DPaddingDataset",
        ann_file=annotation_path,
        subset_name="train",
        block_list=None,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # dataloader setting
        feature_stride=8,
        sample_stride=1,
        offset_frames=4,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="RandomTrunc", trunc_len=window_size, trunc_thresh=0.3, crop_ratio=[0.9, 1.0]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        type="Ego4DSlidingDataset",
        ann_file=annotation_path,
        subset_name="val",
        block_list=None,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # dataloader setting
        window_size=window_size,
        feature_stride=8,
        sample_stride=1,
        offset_frames=4,
        window_overlap_ratio=0,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        type="Ego4DSlidingDataset",
        ann_file=annotation_path,
        subset_name="val",
        block_list=None,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        # dataloader setting
        window_size=window_size,
        feature_stride=8,
        sample_stride=1,
        offset_frames=4,
        window_overlap_ratio=0,
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
    subset="val",
    tiou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
    ground_truth_filename=annotation_path,
    top_k=[1, 5],
)
