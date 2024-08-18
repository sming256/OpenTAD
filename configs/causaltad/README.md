# CausalTAD

> [Harnessing Temporal Causality for Advanced Temporal Action Detection](https://arxiv.org/abs/2407.17792)  
> Shuming Liu, Lin Sui, Chen-Lin Zhang, Fangzhou Mu, Chen Zhao, Bernard Ghanem

<!-- [ALGORITHM] -->

## Abstract

As a fundamental task in long-form video understanding, temporal action detection (TAD) aims to capture inherent temporal relations in untrimmed videos and identify candidate actions with precise boundaries. Over the years, various networks, including convolutions, graphs, and transformers, have been explored for effective temporal modeling for TAD. However, these modules typically treat past and future information equally, overlooking the crucial fact that changes in action boundaries are essentially causal events. 
Inspired by this insight, we propose leveraging the temporal causality of actions to enhance TAD representation by restricting the model's access to only past or future context. We introduce CausalTAD, which combines causal attention and causal Mamba to achieve state-of-the-art performance on multiple benchmarks. Notably, with CausalTAD, we ranked 1st in the Action Recognition, Action Detection, and Audio-Based Interaction Detection tracks at the EPIC-Kitchens Challenge 2024, as well as 1st in the Moment Queries track at the Ego4D Challenge 2024.

## Results and Models

**ActivityNet-1.3**

| Features | Classifier | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |        Config         |                                                                                          Download                                                                                          |
| :------: | :--------: | :-----: | :------: | :------: | :------: | :-------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   TSP    |    CUHK    |  55.62  |  38.51   |   9.40   |  37.46   | [config](anet_tsp.py) | [model](https://drive.google.com/file/d/1U9vnuYllvn-uce1JhOTrOWUfxhHTfMJc/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1bPVe27CHTDFsny4o-BanJzlZf_FWwafx/view?usp=sharing) |


**THUMOS-14**

| Features | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |         Config          |                                                                                          Download                                                                                          |
| :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :---------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   I3D    |  84.43  |  80.75  |  73.57  |  62.70  |  47.33  |  69.75   | [config](thumos_i3d.py) | [model](https://drive.google.com/file/d/19e6JLDO08HUkadvOf8aTaMJ8IapRowv5/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1baXd9ZrlryefMJHEeBZvyKhbLSIY1V7w/view?usp=sharing) |


**Epic-Kitchens-100**

| Subset | Features | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | ave. mAP |             Config              |                                                                                          Download                                                                                          |
| :----: | :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-----------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  Noun  | SlowFast |  28.13  |  26.79  |  25.16  |  22.63  |  18.70  |  24.28   | [config](epic_slowfast_noun.py) | [model](https://drive.google.com/file/d/13wUifnPfDMga7Hi29l33JtDdhlbhKA5k/view?usp=sharing)   \| [log](https://drive.google.com/file/d/17NpOHgCS18otReaJw74rwTx3TlKeAWQJ/view?usp=sharing) |
|  Verb  | SlowFast |  29.62  |  28.69  |  27.16  |  25.24  |  21.44  |  26.43   | [config](epic_slowfast_verb.py) | [model](https://drive.google.com/file/d/1WvbS8YQlb79KFIZWM0Tu-MDl3lETNQd2/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1DluggD1UQ1iz6B19j6L29aXNbhiqDBHC/view?usp=sharing) |

**Ego4D-MQ**

|   Features   | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | ave. mAP |             Config              |                                                                                          Download                                                                                          |
| :----------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-----------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| InternVideo1 |  37.68  |  35.28  |  32.23  |  29.49  |  26.29  |  32.19   | [config](ego4d_internvideo1.py) | [model](https://drive.google.com/file/d/1Uc1ZUJjB9gGdVzC6ej-ek5nlH7SY2VwR/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1za48RI__Ed0DUCHpLp5Wqp3s7BtikIKW/view?usp=sharing) |
| InternVideo2 |  39.01  |  36.05  |  33.06  |  30.45  |  26.70  |  33.05   | [config](ego4d_internvideo2.py) | [model](https://drive.google.com/file/d/101--h23mBx7F3B8ezAabbuKlRbwB9vpj/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1A5oSrmh-kzF6vQr5J-maiUIB34ZsSk5o/view?usp=sharing) |

For our solution to Ego4D Challenge 2024 and EPIC-Kitchens Challenge 2024, please refer to [here](egovis_challenge_2024/README.md), including detailed challenge config and ensemble strategy.


## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train CausalTAD on THUMOS dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/causaltad/thumos_i3d.py
```

For more details, you can refer to the Training part in the [Usage](../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test CausalTAD on THUMOS dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/causaltad/thumos_i3d.py --checkpoint exps/thumos/causal_i3d/gpu1_id0/checkpoint/epoch_38.pth
```

For more details, you can refer to the Test part in the [Usage](../../docs/en/usage.md).

## Citation

```latex
@article{liu2024harnessing,
  title={Harnessing Temporal Causality for Advanced Temporal Action Detection},
  author={Liu, Shuming and Sui, Lin and Zhang, Chen-Lin and Mu, Fangzhou and Zhao, Chen and Ghanem, Bernard},
  journal={arXiv preprint arXiv:2407.17792},
  year={2024}
}
```
