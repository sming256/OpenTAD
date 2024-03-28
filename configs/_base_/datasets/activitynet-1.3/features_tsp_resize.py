dataset_type = "AnetResizeDataset"
annotation_path = "data/activitynet-1.3/annotations/activity_net.v1-3.min.json"
class_map = "data/activitynet-1.3/annotations/category_idx.txt"
data_path = "data/activitynet-1.3/features/anet_tsp_npy_unresize/"
block_list = "data/activitynet-1.3/features/anet_tsp_npy_unresize/missing_files.txt"

resize_length = 128

dataset = dict(
    train=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="training",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=True,
        resize_length=resize_length,
        class_agnostic=True,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy", prefix="v_"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="ResizeFeat", tool="torchvision_align"),
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
        resize_length=resize_length,
        class_agnostic=True,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy", prefix="v_"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="ResizeFeat", tool="torchvision_align"),
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
        resize_length=resize_length,
        class_agnostic=True,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy", prefix="v_"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="ResizeFeat", tool="torchvision_align"),
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
