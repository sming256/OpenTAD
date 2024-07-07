_base_ = [
    "../_base_/datasets/thumos-14/features_i3d_pad.py",  # dataset config
    "../_base_/models/dyfadet.py",  # model config
]

data_path = "data/thumos-14/features/videomaev2-giant_k710_16x4x1_img224_stride4_len16_interval1_thumos/"
dataset = dict(
    train=dict(data_path=data_path, offset_frames=2),
    val=dict(data_path=data_path, offset_frames=2),
    test=dict(data_path=data_path, offset_frames=2),
)

model = dict(projection=dict(in_channels=1408, input_noise=1e-5, path_pdrop=0.005))

solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=1, num_workers=1),
    test=dict(batch_size=1, num_workers=1),
    clip_grad_norm=1,
    ema=True,
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.025, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=20, max_epoch=60)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=2000,
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.7,  #  set 0 to disable
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=20,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=30,
    end_epoch=45,
)

work_dir = "exps/thumos/dyfadet_videomaev2_g"
