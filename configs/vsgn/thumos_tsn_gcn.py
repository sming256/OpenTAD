_base_ = ["thumos_tsn.py"]

model = dict(
    projection=dict(
        use_gcn=True,
        gcn_kwargs=dict(num_neigh=12, nfeat_mode="feat_ctr", agg_type="max", edge_weight="false"),
    ),
)

work_dir = "exps/thumos/vsgn_tsn_sw1280_gcn"
