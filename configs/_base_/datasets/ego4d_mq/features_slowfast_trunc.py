dataset_type = "Ego4DPaddingDataset"
annotation_path = "data/ego4d/annotations/ego4d_v2_220429_feature.json"
class_map = "data/ego4d/annotations/category_idx.txt"
data_path = "data/ego4d/features/mq_slowfast/"
block_list = None

trunc_len = 1024

dataset = dict(
    train=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="train",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=True,
        # thumos dataloader setting
        feature_stride=16,
        sample_stride=1,
        offset_frames=16,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="RandomTrunc", trunc_len=trunc_len, trunc_thresh=0.3, crop_ratio=[0.9, 1.0]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="val",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # thumos dataloader setting
        feature_stride=16,
        sample_stride=1,
        offset_frames=16,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="val",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        # thumos dataloader setting
        feature_stride=16,
        sample_stride=1,
        offset_frames=16,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats"]),
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
