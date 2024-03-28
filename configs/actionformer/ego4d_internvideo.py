_base_ = ["ego4d_slowfast.py"]

model = dict(projection=dict(in_channels=1024))

work_dir = "exps/ego4d/actionformer_internvideo"
