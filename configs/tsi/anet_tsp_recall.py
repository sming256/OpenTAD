_base_ = ["anet_tsp.py"]

evaluation = dict(
    type="Recall",
    subset="validation",
    topk=[1, 5, 10, 100],
    max_avg_nr_proposals=100,
    tiou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
)

post_processing = dict(
    score_type="iou*s*e",
    proposal=True,
    nms=dict(
        use_soft_nms=True,
        sigma=0.35,
        max_seg_num=100,
        min_score=0,
        multiclass=False,
        voting_thresh=0.95,  #  set 0 to disable
        method=3,
        t1=0.55,
        t2=0.37,
    ),
    external_cls=dict(_delete_=True, type="PseudoClassifier"),
    save_dict=False,
)
