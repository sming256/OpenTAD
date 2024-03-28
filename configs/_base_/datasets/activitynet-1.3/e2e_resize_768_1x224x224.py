dataset_type = "AnetResizeDataset"
annotation_path = "data/activitynet-1.3/annotations/activity_net.v1-3.min.json"
class_map = "data/activitynet-1.3/annotations/category_idx.txt"
data_path = "data/activitynet-1.3/raw_data/Anet_videos_15fps_short256"
block_list = "data/activitynet-1.3/raw_data/Anet_videos_15fps_short256/missing_files.txt"

resize_length = 768

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
            dict(type="PrepareVideoInfo", format="mp4", prefix="v_"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="resize"),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 224)),
            dict(type="mmaction.CenterCrop", crop_size=224),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
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
            dict(type="PrepareVideoInfo", format="mp4", prefix="v_"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="resize"),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 224)),
            dict(type="mmaction.CenterCrop", crop_size=224),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
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
            dict(type="PrepareVideoInfo", format="mp4", prefix="v_"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="resize"),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 224)),
            dict(type="mmaction.CenterCrop", crop_size=224),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
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
