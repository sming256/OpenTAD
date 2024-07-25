annotation_path = "data/epic_kitchens-100/annotations/epic_kitchens_noun.json"
class_map = "data/epic_kitchens-100/annotations/category_idx_noun.txt"
data_path = (
    "data/epic_kitchens-100/features/internvideo2_1B_ft_k710_ft_k700_ft_noun_img224_stride8_len16_interval1_epic/"
)
block_list = None

window_size = 4608

dataset = dict(
    train=dict(
        type="EpicKitchensPaddingDataset",
        ann_file=annotation_path,
        subset_name="train",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=True,
        # epic-kitchens dataloader setting
        fps=30,
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
        type="EpicKitchensSlidingDataset",
        ann_file=annotation_path,
        subset_name="val",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # dataloader setting
        window_size=window_size,
        fps=30,
        feature_stride=8,
        sample_stride=1,
        offset_frames=4,
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
        subset_name="val",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        # epic-kitchens dataloader setting
        window_size=window_size,
        fps=30,
        feature_stride=8,
        sample_stride=1,
        offset_frames=4,
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
    subset="val",
    tiou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
    ground_truth_filename=annotation_path,
)
