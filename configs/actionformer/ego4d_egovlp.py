_base_ = ["ego4d_slowfast.py"]

data_path = "data/ego4d/features/mq_egovlp/"
dataset = dict(
    train=dict(data_path=data_path, offset_frames=8),
    val=dict(data_path=data_path, offset_frames=8),
    test=dict(data_path=data_path, offset_frames=8),
)

model = dict(
    projection=dict(in_channels=256, out_channels=384),
    neck=dict(in_channels=384, out_channels=384),
    rpn_head=dict(in_channels=384, feat_channels=384),
)

work_dir = "exps/ego4d/actionformer_egovlp"
