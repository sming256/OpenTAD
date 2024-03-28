dataset_type = "EpicKitchensPaddingDataset"
annotation_path = "data/charades/annotations/charades.json"
class_map = "data/charades/annotations/category_idx.txt"
data_path = "data/charades/features/i3d_charades_finetuned_stride8/"
block_list = data_path + "missing_files.txt"

trunc_len = 768

dataset = dict(
    train=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="training",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # dataloader setting
        fps=24,
        feature_stride=8,
        sample_stride=1,
        offset_frames=0,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="RandomTrunc", trunc_len=trunc_len, trunc_thresh=0.5, crop_ratio=[0.9, 1.0]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="testing",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # dataloader setting
        fps=24,
        feature_stride=8,
        sample_stride=1,
        offset_frames=0,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="Padding", length=trunc_len),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="testing",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        # dataloader setting
        fps=24,
        feature_stride=8,
        sample_stride=1,
        offset_frames=0,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="Padding", length=trunc_len),
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
