# TemporalMaxer

> [TemporalMaxer: Maximize Temporal Context with only Max Pooling for Temporal Action Localization](https://arxiv.org/abs/2303.09055)  
> Tuan N. Tang, Kwonyoung Kim, Kwanghoon Sohn

<!-- [ALGORITHM] -->

## Abstract

Temporal Action Localization (TAL) is a challenging task in video understanding that aims to identify and localize actions within a video sequence. Recent studies have emphasized the importance of applying long-term temporal context modeling (TCM) blocks to the extracted video clip features such as employing complex self-attention mechanisms. In this paper, we present the simplest method ever to address this task and argue that the extracted video clip features are already informative to achieve outstanding performance without sophisticated architectures. To this end, we introduce TemporalMaxer, which minimizes long-term temporal context modeling while maximizing information from the extracted video clip features with a basic, parameter-free, and local region operating max-pooling block. Picking out only the most critical information for adjacent and local clip embeddings, this block results in a more efficient TAL model. We demonstrate that TemporalMaxer outperforms other state-of-the-art methods that utilize long-term TCM such as self-attention on various TAL datasets while requiring significantly fewer parameters and computational resources. The code for our approach is publicly available at https://github.com/TuanTNG/TemporalMaxer.

## Results and Models

**Note:** TemporalMaxer needs more than 40 GB GPU memory when training on Multi-THUMOS and Epic-kitchen.

**THUMOS-14**

| Features | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |         Config          |                                                                                          Download                                                                                          |
| :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :---------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   I3D    |  83.17  |  79.09  |  71.66  |  61.72  |  46.00  |  68.33   | [config](thumos_i3d.py) | [model](https://drive.google.com/file/d/1Qh1CBphRbU0R07FRLk1nkQ_g53na6WLv/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1jQPUKp4X1amda1NJ2AuzK1SDxiT7pbxc/view?usp=sharing) |

**MultiTHUMOS**

|    Features    | mAP@0.2 | mAP@0.5 | mAP@0.7 | ave. mAP (0.1:0.9:0.1) |              Config              |                                                                                          Download                                                                                          |
| :------------: | :-----: | :-----: | :-----: | :--------------------: | :------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   I3D (rgb)    |  53.57  |  38.39  |  20.61  |         33.92          | [config](multithumos_i3d_rgb.py) | [model](https://drive.google.com/file/d/1xci9meo5Gb0XMUTo2fUt3d-sxBUlWefZ/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1oQKds1GFERu54nBdNBxRY_1wUnRlIrXy/view?usp=sharing) |
| I3D (rgb+flow) |  60.05  |  44.86  |  25.59  |         39.26          |   [config](multithumos_i3d.py)   | [model](https://drive.google.com/file/d/1rfIDpTE0E2VOi4b4CpndlEntSdrnrEC2/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1pJfpR7Ssy2pucZSbQfLhbqHamwpzcnLW/view?usp=sharing) |

**Epic-Kitchens-100**

| Subset | Features | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | ave. mAP |                  Config                  |                                                                                          Download                                                                                          |
| :----: | :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :--------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  Noun  | SlowFast |  25.46  |  24.35  |  22.55  |  20.32  |  17.08  |  21.96   | [config](epic_kitchens_slowfast_noun.py) | [model](https://drive.google.com/file/d/17Iuc5RHMje1BW8XnECIkeKX7q_BNvNXo/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1taY9F7x2avb0jTRsiGdwCNepuYQ9wi9u/view?usp=sharing) |
|  Verb  | SlowFast |  28.64  |  27.84  |  25.59  |  23.60  |  19.69  |  25.07   | [config](epic_kitchens_slowfast_verb.py) | [model](https://drive.google.com/file/d/1OaECxAV_HS6WBTb30tZFFczhLziRVYdW/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1LUR5vfzAcTW5WzDB1u3GPZgOfQ0HOF_3/view?usp=sharing) |


## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TemporalMaxer on THUMOS dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/temporalmaxer/thumos_i3d.py
```

For more details, you can refer to the Training part in the [Usage](../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TemporalMaxer on THUMOS dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/temporalmaxer/thumos_i3d.py --checkpoint exps/thumos/temporalmaxer_i3d/gpu1_id0/checkpoint/epoch_45.pth
```

For more details, you can refer to the Test part in the [Usage](../../docs/en/usage.md).

## Citation

```latex
@article{tang2023temporalmaxer,
  title={TemporalMaxer: Maximize Temporal Context with only Max Pooling for Temporal Action Localization},
  author={Tang, Tuan N and Kim, Kwonyoung and Sohn, Kwanghoon},
  journal={arXiv preprint arXiv:2303.09055},
  year={2023}
}
```