_base_ = ["e2e_charades_videomae_s_512x1_160_adapter.py"]

model = dict(
    backbone=dict(
        backbone=dict(embed_dims=768, depth=12, num_heads=12),
        custom=dict(pretrain="pretrained/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth"),
    ),
    projection=dict(in_channels=768),
)

optimizer = dict(backbone=dict(custom=[dict(name="adapter", lr=4e-4, weight_decay=0.05)]))

work_dir = "exps/charades/adatad/e2e_actionformer_videomae_b_512x1_160_adapter"
