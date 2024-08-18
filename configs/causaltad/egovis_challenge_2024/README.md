

# Solution to Ego4D Moment Retrieval and EPIC-Kitchens Challenge 2024


## Ego4D-MQ Challenge

**STEP1:** train the CausalTAD model (InternVideo1+InternVideo2 feature) with 3 different seeds
```
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/causaltad/egovis_challenge_2024/ego4d_internvideo1_internvideo2.py --id 0 --seed 42
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/causaltad/egovis_challenge_2024/ego4d_internvideo1_internvideo2.py --id 1 --seed 288
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/causaltad/egovis_challenge_2024/ego4d_internvideo1_internvideo2.py --id 2 --seed 787878
```
**STEP2:** infer each model (with epoch 14th checkpoint) and save the raw predictions.
```
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/causaltad/egovis_challenge_2024/ego4d_internvideo1_internvideo2.py --cfg-options inference.save_raw_prediction=True --checkpoint exps/ego4d/causal_internvideo1_internvideo2/gpu1_id0/checkpoint/epoch_14.pth --id 0 --not_eval
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/causaltad/egovis_challenge_2024/ego4d_internvideo1_internvideo2.py --cfg-options inference.save_raw_prediction=True --checkpoint exps/ego4d/causal_internvideo1_internvideo2/gpu1_id1/checkpoint/epoch_14.pth --id 1 --not_eval
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/causaltad/egovis_challenge_2024/ego4d_internvideo1_internvideo2.py --cfg-options inference.save_raw_prediction=True --checkpoint exps/ego4d/causal_internvideo1_internvideo2/gpu1_id2/checkpoint/epoch_14.pth --id 2 --not_eval
```

**STEP3:** ensemble the predictions and check the results on validation set. You should get the average mAP around 34.13 on validation set.
```
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/causaltad/egovis_challenge_2024/ego4d_ensemble.py
```

Results on the validation set are as follows:
|         Features          | Ensemble | ave. mAP |                    Config                    |                                                                                          Download                                                                                          |
| :-----------------------: | :------: | :------: | :------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|       InternVideo1        |    No    |  32.19   |      [config](../ego4d_internvideo1.py)      | [model](https://drive.google.com/file/d/1Uc1ZUJjB9gGdVzC6ej-ek5nlH7SY2VwR/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1za48RI__Ed0DUCHpLp5Wqp3s7BtikIKW/view?usp=sharing) |
|       InternVideo2        |    No    |  33.05   |      [config](../ego4d_internvideo2.py)      | [model](https://drive.google.com/file/d/101--h23mBx7F3B8ezAabbuKlRbwB9vpj/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1A5oSrmh-kzF6vQr5J-maiUIB34ZsSk5o/view?usp=sharing) |
| InternVideo1+InternVideo2 |    No    |  33.59   | [config](ego4d_internvideo1_internvideo2.py) | [model](https://drive.google.com/file/d/1D9P3_eDixdRkwHKbxaPx7CO70vSw8thO/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1S54Ni8-EMhP-yZAWC0h7kW5yYp6ZJYbA/view?usp=sharing) |
| InternVideo1+InternVideo2 |   Yes    |  34.13   |         [config](ego4d_ensemble.py)          |                                                 [log](https://drive.google.com/file/d/1GGn2L-7fKkw2brgGWbe-qRBt5XLfwOP_/view?usp=sharing)                                                  |

**STEP4:** submit to [ego4d server](https://eval.ai/web/challenges/challenge-page/1626/leaderboard/3913)

To achieve higher performance on test set, you can train the model with training and validation data, and submit the final prediction to ego4d server.

1. Repeat the STEP1 and STEP2 using config `ego4d_internvideo1_internvideo2_trainval.py`, and infer on test set.
2. Ensemble the prediction and generate the final json file.
```
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/prepare_data/ego4d/ego4d_challenge_submit.py configs/causaltad/egovis_challenge_2024/ego4d_ensemble_trainval.py
```

You should get the average mAP around 34.9\% on the test set.

## EPIC-Kitchens Action Detection Challenge

**STEP1:** train the noun model and verb model separately
```
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/causaltad/egovis_challenge_2024/epic_internvideo2_1b_noun.py
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/causaltad/egovis_challenge_2024/epic_internvideo2_1b_verb.py
```

**STEP2:** check the action/noun/verb results on validation set using joint evaluation metric
```
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/prepare_data/epic/epic_action_noun_verb_submit.py \
    configs/causaltad/egovis_challenge_2024/epic_internvideo2_1b_noun.py \
    configs/causaltad/egovis_challenge_2024/epic_internvideo2_1b_verb.py \
    exps/epic_kitchens/causal_internvideo2_1b_noun/gpu1_id0/checkpoint/epoch_32.pth \
    exps/epic_kitchens/causal_internvideo2_1b_verb/gpu1_id0/checkpoint/epoch_29.pth \
    --pre_nms_topk 5000 --max_seg_num 3000
```

Results on the validation set are as follows:

| Evaluation |   Features   | mAP on Noun | mAP on Verb | mAP on Action |                                   Config                                   |                                                                                         Model                                                                                          |                                                                                          Log                                                                                          |
| :--------: | :----------: | :---------: | :---------: | :-----------: | :------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  Separate  | InternVideo2 |    36.25    |    32.91    |       -       | [noun](epic_internvideo2_1b_noun.py), [verb](epic_internvideo2_1b_verb.py) | [noun](https://drive.google.com/file/d/1ntQ5DJwVgUOsAcoJNvohjOfCQ07XtHn5/view?usp=sharing), [verb](https://drive.google.com/file/d/1ubbTwumQEJ-79CH2d_D7ymbZgw0DoOeP/view?usp=sharing) | [noun](https://drive.google.com/file/d/1dnSbMsHZ4byCBV8CbuyF9sB67lTrk__R/view?usp=sharing),[verb](https://drive.google.com/file/d/1rVdMkUdDPd5rUsBGwhJPFVODrAnflcbu/view?usp=sharing) |
|   Joint    | InternVideo2 |    34.72    |    26.87    |     28.86     |                                     -                                      |                                                                                         above                                                                                          |                                               [log](https://drive.google.com/file/d/1iJ3mNSSJHwcEbRXY55oZK5JJHnVbIf7_/view?usp=sharing)                                               |

**STEP3:** submit to [epic kitchens server](https://codalab.lisn.upsaclay.fr/competitions/707)

To achieve higher performance on test set, you can train the model with training and validation data, and submit the final prediction to epic-kitchens server.

1. Repeat the STEP1 using config `epic_internvideo2_1b_noun_trainval.py`, `epic_internvideo2_1b_verb_trainval.py`.
2. Generate the final json file.
```
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/prepare_data/epic/epic_action_noun_verb_submit.py \
    configs/causaltad/egovis_challenge_2024/epic_internvideo2_1b_noun_trainval.py \
    configs/causaltad/egovis_challenge_2024/epic_internvideo2_1b_verb_trainval.py \
    exps/epic_kitchens/causal_internvideo2_1b_noun_trainval/gpu1_id0/checkpoint/epoch_32.pth \
    exps/epic_kitchens/causal_internvideo2_1b_verb_trainval/gpu1_id0/checkpoint/epoch_29.pth \
    --submit
```

You should get the average mAP around 31.9\% of the action task on the test set.

## EPIC-Kitchens Audio-Based Interaction Detection

**STEP1:** train the CausalTAD model with [Auditory-SlowFast features](https://github.com/epic-kitchens/C10-epic-sounds-detection?tab=readme-ov-file#features)
```
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/causaltad/egovis_challenge_2024/epic_sounds_audioslowfast.py
```

Results on the validation set are as follows:

|     Features      | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | ave. mAP |                 Config                 |                                                                                          Download                                                                                          |
| :---------------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Auditory-SlowFast |  16.98  |  15.73  |  14.72  |  12.35  |  10.44  |  14.04   | [config](epic_sounds_audioslowfast.py) | [model](https://drive.google.com/file/d/1xcnmfCFa4ZABKgICJM5itH-sComRoOVE/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1STy8tZmF50dk9JpR2RDxZTuCq5-FaICG/view?usp=sharing) |


**STEP2:** submit to [epic kitchens server](https://codalab.lisn.upsaclay.fr/competitions/17921)

To achieve higher performance on test set, you can train the model with training and validation data, and submit the final prediction to epic-kitchens server.

1. Repeat the STEP1 using config `epic_sounds_audioslowfast_trainval.py`.
2. Generate the final json file.
```
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/prepare_data/epic_sounds/epic_sound_submit.py \
    configs/causaltad/egovis_challenge_2024/epic_sounds_audioslowfast_trainval.py \
    exps/epic_sounds/causal_audioslowfast_trainval/gpu1_id0/checkpoint/epoch_16.pth
```

You should get the average mAP around 14.8\% on test set.