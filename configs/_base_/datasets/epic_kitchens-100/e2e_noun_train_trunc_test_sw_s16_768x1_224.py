annotation_path = "data/epic_kitchens-100/annotations/epic_kitchens_noun.json"
class_map = "data/epic_kitchens-100/annotations/category_idx_noun.txt"
data_path = "data/epic_kitchens-100/raw_data/epic_kitchens_100_30fps_512x288/"
block_list = None

window_size = 768

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
        feature_stride=16,
        sample_stride=1,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=window_size,
                trunc_thresh=0.3,
                crop_ratio=[0.9, 1.0],
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 256)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(224, 224), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
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
        feature_stride=16,
        sample_stride=1,
        window_overlap_ratio=0.25,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window"),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 224)),
            dict(type="mmaction.CenterCrop", crop_size=224),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
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
        feature_stride=16,
        sample_stride=1,
        window_overlap_ratio=0.5,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window"),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 224)),
            dict(type="mmaction.CenterCrop", crop_size=224),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
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
