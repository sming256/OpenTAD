

# Solution to Ego4D Moment Retrieval and EPIC-Kitchens Action Detection Challenge 2024


## Ego4D-MQ Challenge

**STEP1:** train the model with 3 different seeds
```
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/causaltad/egovis_challenge_2024/ego4d_internvideo1_internvideo2.py --id 0 --seed 42
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/causaltad/egovis_challenge_2024/ego4d_internvideo1_internvideo2.py --id 1 --seed 288
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/causaltad/egovis_challenge_2024/ego4d_internvideo1_internvideo2.py --id 2 --seed 787878
```
**STEP2:** infer each experiments (with epoch 14th checkpoint) and save the predictions
```
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py  configs/causaltad/egovis_challenge_2024/ego4d_internvideo1_internvideo2.py --cfg-options inference.save_raw_prediction=True  --checkpoint exps/ego4d/causal_internvideo2_trainval/gpu1_id0/checkpoint/epoch_14.pth --id 0 --not_eval
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py  configs/causaltad/egovis_challenge_2024/ego4d_internvideo1_internvideo2.py --cfg-options inference.save_raw_prediction=True  --checkpoint exps/ego4d/causal_internvideo2_trainval/gpu1_id1/checkpoint/epoch_14.pth --id 1 --not_eval
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py  configs/causaltad/egovis_challenge_2024/ego4d_internvideo1_internvideo2.py --cfg-options inference.save_raw_prediction=True  --checkpoint exps/ego4d/causal_internvideo2_trainval/gpu1_id2/checkpoint/epoch_14.pth --id 2 --not_eval
```

**STEP3:** ensemble the predictions and check the results on validation set
```
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/causaltad/egovis_challenge_2024/ego4d_ensemble.py   
```

|         Features          | Ensemble | ave. mAP |                    Config                    |        Download        |
| :-----------------------: | :------: | :------: | :------------------------------------------: | :--------------------: |
|       InternVideo1        |    No    |  32.19   |      [config](../ego4d_internvideo1.py)      | [model]()   \| [log]() |
|       InternVideo2        |    No    |  33.05   |       [config](ego4d_internvideo2.py)        | [model]()   \| [log]() |
| InternVideo1+InternVideo2 |    No    |  33.59   | [config](ego4d_internvideo1_internvideo2.py) | [model]()   \| [log]() |
| InternVideo1+InternVideo2 |   Yes    |  34.21   | [config](ego4d_internvideo1_internvideo2.py) | [model]()   \| [log]() |

**STEP4:** submit to ego4d server
1. Repeat the STEP1 and STEP2 using config with training and validation data. (ego4d_internvideo1_internvideo2_trainval.py)
2. Generate the final json file.
```
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/prepare_data/ego4d/ego4d_challenge_submit.py configs/causaltad/egovis_challenge_2024/ego4d_ensemble_trainval.py
```

## EPIC-Kitchens Action Detection Challenge

**STEP1:** train the noun model and verb model separately
```
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py  configs/causaltad/egovis_challenge_2024/epic_internvideo2_1b_noun.py
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py  configs/causaltad/egovis_challenge_2024/epic_internvideo2_1b_verb.py
```

**STEP2:** check the action/noun/verb results on validation set using joint evaluation metric
```
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/prepare_data/epic/epic_action_noun_verb_submit.py onfigs/causaltad/egovis_challenge_2024/epic_internvideo2_1b_noun.py configs/causaltad/egovis_challenge_2024/epic_internvideo2_1b_verb.py exps/epic_kitchens/causal_internvideo2_1b_noun/gpu1_id0/checkpoint/epoch_32.pth exps/epic_kitchens/causal_internvideo2_1b_verb/gpu1_id0/checkpoint/epoch_29.pth
```

**STEP3:** submit to epic kitchens server
1. Repeat the STEP1 using config with training and validation data. (epic_internvideo2_1b_noun_trainval.py, epic_internvideo2_1b_verb_trainval.py)
2. Generate the final json file.
```
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/prepare_data/epic/epic_action_noun_verb_submit.py onfigs/causaltad/egovis_challenge_2024/epic_internvideo2_1b_noun_trainval.py configs/causaltad/egovis_challenge_2024/epic_internvideo2_1b_verb_trainval.py exps/epic_kitchens/causal_internvideo2_1b_noun_trainval/gpu1_id0/checkpoint/epoch_32.pth exps/epic_kitchens/causal_internvideo2_1b_verb_trainval/gpu1_id0/checkpoint/epoch_29.pth
--submit
```
