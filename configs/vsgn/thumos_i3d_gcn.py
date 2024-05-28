_base_ = ["thumos_i3d.py"]

model = dict(
    projection=dict(
        use_gcn=True,
        gcn_kwargs=dict(num_neigh=10, nfeat_mode="feat_ctr", agg_type="max", edge_weight="false"),
    ),
)

work_dir = "exps/thumos/vsgn_i3d_gcn"
