_base_ = [
    "../_base_/datasets/activitynet-1.3/e2e_resize_128_16x128x128.py",  # dataset config
    "../_base_/models/etad.py",  # model config
]

model = dict(
    backbone=dict(
        type="mmaction.Recognizer3D",
        backbone=dict(
            type="ResNet2Plus1d_TSP",
            layers=[3, 4, 6, 3],
            pretrained="pretrained/r2plus1d_34-tsp_on_activitynet-max_gvf-backbone_lr_0.0001-fc_lr_0.002-epoch_5-0d2cf854.pth",
            frozen_stages=2,
            norm_eval=True,
        ),
        data_preprocessor=dict(
            type="mmaction.ActionDataPreprocessor",
            mean=[110.2008, 100.63983, 95.99475],
            std=[58.14765, 56.46975, 55.332195],
            format_shape="NCTHW",
        ),
        custom=dict(
            pretrained="pretrained/r2plus1d_34-tsp_on_activitynet-max_gvf-backbone_lr_0.0001-fc_lr_0.002-epoch_5-0d2cf854.pth",
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b n c", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="b n c -> b c n"),
            ],
            norm_eval=True,  # also update the norm layers
            freeze_backbone=False,  # unfreeze the backbone
            temporal_checkpointing=True,  # enable temporal checkpointing, default is False
            temporal_checkpointing_chunk_num=16,  #  number of chunks to split the temporal dimension
            temporal_checkpointing_chunk_dim=0,  # input shape is [B*N,3,T,H,W], split the B*N dimension
        ),
    ),
    projection=dict(in_channels=512),
)

solver = dict(
    train=dict(batch_size=4, num_workers=4),
    val=dict(batch_size=4, num_workers=4),
    test=dict(batch_size=4, num_workers=4),
    clip_grad_norm=1,
)

optimizer = dict(
    type="AdamW",
    lr=5e-4,
    weight_decay=1e-4,
    backbone=dict(lr=5e-7, weight_decay=1e-4),
)
scheduler = dict(type="MultiStepLR", milestones=[5], gamma=0.1, max_epoch=6)

inference = dict(test_epoch=5, load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.35,
        max_seg_num=100,
        min_score=0.0001,
        multiclass=False,
        voting_thresh=0,  #  set 0 to disable
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
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=5,
)

work_dir = "exps/anet/e2e_etad_tsp_128_ratio0.3"
