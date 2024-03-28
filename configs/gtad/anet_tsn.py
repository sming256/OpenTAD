_base_ = [
    "../_base_/datasets/activitynet-1.3/features_tsp_resize.py",  # dataset config
    "../_base_/models/gtad.py",  # model config
]

resize_length = 100
data_path = "data/activitynet-1.3/features/anet_tsn_npy_unresize/"
block_list = "data/activitynet-1.3/features/anet_tsn_npy_unresize/missing_files.txt"
dataset = dict(
    train=dict(data_path=data_path, block_list=block_list, resize_length=resize_length),
    val=dict(data_path=data_path, block_list=block_list, resize_length=resize_length),
    test=dict(data_path=data_path, block_list=block_list, resize_length=resize_length),
)

solver = dict(
    train=dict(batch_size=16, num_workers=4),
    val=dict(batch_size=16, num_workers=4),
    test=dict(batch_size=16, num_workers=4),
    clip_grad_norm=1,
)

optimizer = dict(type="Adam", lr=4e-3, weight_decay=1e-4)
scheduler = dict(type="MultiStepLR", milestones=[5], gamma=0.1, max_epoch=10)


inference = dict(test_epoch=7, load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.8,  # 0.45, 0.55
        max_seg_num=100,
        iou_threshold=0,  # does not matter when use soft nms
        voting_thresh=0.95,  #  set 0 to disable
    ),
    external_cls=dict(
        type="CUHKANETClassifier",
        path="data/activitynet-1.3/classifiers/cuhk_val_simp_7.json",
        topk=2,
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=200,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=4,
)

work_dir = "exps/anet/gtad_tsn_100x100"
