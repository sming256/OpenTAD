# ActionFormer

> [ActionFormer: Localizing Moments of Actions with Transformers](https://arxiv.org/abs/2202.07925)  
> Chen-Lin Zhang, Jianxin Wu, Yin Li

<!-- [ALGORITHM] -->

## Abstract

Self-attention based Transformer models have demonstrated impressive results for image classification and object detection, and more recently for video understanding. Inspired by this success, we investigate the application of Transformer networks for temporal action localization in videos. To this end, we present ActionFormer -- a simple yet powerful model to identify actions in time and recognize their categories in a single shot, without using action proposals or relying on pre-defined anchor windows. ActionFormer combines a multiscale feature representation with local self-attention, and uses a light-weighted decoder to classify every moment in time and estimate the corresponding action boundaries. We show that this orchestrated design results in major improvements upon prior works. Without bells and whistles, ActionFormer achieves 71.0% mAP at tIoU=0.5 on THUMOS14, outperforming the best prior model by 14.1 absolute percentage points. Further, ActionFormer demonstrates strong results on ActivityNet 1.3 (36.6% average mAP) and EPIC-Kitchens 100 (+13.5% average mAP over prior works).

## Results and Models

**ActivityNet-1.3** with CUHK classifier.

| Features | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |        Config         |                                                                                          Download                                                                                          |
| :------: | :-----: | :------: | :------: | :------: | :-------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   TSP    |  55.08  |  38.27   |   8.91   |  37.07   | [config](anet_tsp.py) | [model](https://drive.google.com/file/d/1loC72F4U79jWfoRL9SB2rdk3xykBKqHN/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1YveGerbI1es51t2Ii7WZDgPlJy3lGBLf/view?usp=sharing) |

**THUMOS-14**

| Features | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |         Config          |                                                                                          Download                                                                                          |
| :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :---------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   I3D    |  83.78  |  80.06  |  73.16  |  60.46  |  44.72  |  68.44   | [config](thumos_i3d.py) | [model](https://drive.google.com/file/d/17oP-fMOjw6wwnaQWTlikWwoZoSkiIFkt/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1WJe98mKoXaP2X9Th-gKC8rw0JeKxfJkq/view?usp=sharing) |

**HACS**

| Features | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |           Config           |                                                                                          Download                                                                                          |
| :------: | :-----: | :------: | :------: | :------: | :------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| SlowFast |  56.18  |  37.97   |  11.05   |  37.71   | [config](hacs_slowfast.py) | [model](https://drive.google.com/file/d/1IdxR5lyfXzk5wjl-8YDcH0Nw2BEDwzWz/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1Eu2O9IKuR8XLeZ37OxCq7NjSUKPE-3Zw/view?usp=sharing) |

**Epic-Kitchens-100**

| Subset | Features | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | ave. mAP |                  Config                  |                                                                                          Download                                                                                          |
| :----: | :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :--------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  Noun  | SlowFast |  25.78  |  24.73  |  22.83  |  20.84  |  17.45  |  22.33   | [config](epic_kitchens_slowfast_noun.py) | [model](https://drive.google.com/file/d/1RckzXf5W8oD_ARZw5dyYo03ZKVrU1n9-/view?usp=sharing)   \| [log](https://drive.google.com/file/d/18dVA27hWRBjM8lp4S12DscCkNJBqFrWp/view?usp=sharing) |
|  Verb  | SlowFast |  27.68  |  26.79  |  25.62  |  24.06  |  20.48  |  24.93   | [config](epic_kitchens_slowfast_verb.py) | [model](https://drive.google.com/file/d/1-RLtnku727Fh39rihyGVxLCU5klTIvbn/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1w18Ccyi22ZHgM0ECx6rAKOXqFoO9L0Iq/view?usp=sharing) |

**Ego4D-MQ**

|  Features   | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | ave. mAP |             Config             |                                                                                          Download                                                                                          |
| :---------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :----------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  SlowFast   |  20.90  |  18.12  |  15.81  |  14.25  |  12.21  |  16.26   |  [config](ego4d_slowfast.py)   | [model](https://drive.google.com/file/d/1QMzpP281_XAz5woGmiLercBaUGVrdrv0/view?usp=sharing)   \| [log](https://drive.google.com/file/d/16oRGGq7LiiYCv7yeqG9TR2f6bS-3Fi6r/view?usp=sharing) |
|   EgoVLP    |  27.79  |  24.97  |  22.37  |  19.25  |  16.25  |  22.13   |   [config](ego4d_egovlp.py)    | [model](https://drive.google.com/file/d/1c23BHCCuy7bOlyRkwXMSnTyeA3jRedGt/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1Gu0uaW6ICcarL_wLcwiQTTSf6NqyQMjQ/view?usp=sharing) |
| InternVideo |  32.59  |  30.28  |  27.53  |  25.09  |  22.13  |  27.52   | [config](ego4d_internvideo.py) | [model](https://drive.google.com/file/d/1Q25ZxXIlSi6vr5T4EX4Z75_Iq1uU8Pdi/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1_7Xm3_1Qg0MLXGW5Wx6XyfTTc0Pnn0dt/view?usp=sharing) |


**MultiTHUMOS**

|    Features    | mAP@0.2 | mAP@0.5 | mAP@0.7 | ave. mAP (0.1:0.9:0.1) |              Config              |                                                                                          Download                                                                                          |
| :------------: | :-----: | :-----: | :-----: | :--------------------: | :------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   I3D (rgb)    |  53.52  |  39.05  |  19.69  |         34.02          | [config](multithumos_i3d_rgb.py) | [model](https://drive.google.com/file/d/1iZ1KeXDQJFLFKe24bCSIoJv6DmtBebSM/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1Z2Or_d1aarXr7mjbVVJMnzsI5ieDlLed/view?usp=sharing) |
| I3D (rgb+flow) |  60.18  |  45.01  |  24.56  |         39.19          |   [config](multithumos_i3d.py)   | [model](https://drive.google.com/file/d/1GvaxJdZhL01DYIWg3QWA5BXdm_rLor32/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1KGLz30nbBtwv235tZyPkr2SPvz79koq9/view?usp=sharing) |

**Charades**

| Features  | mAP@0.2 | mAP@0.5 | mAP@0.7 | ave. mAP (0.1:0.9:0.1) |            Config             |                                                                                          Download                                                                                          |
| :-------: | :-----: | :-----: | :-----: | :--------------------: | :---------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| I3D (rgb) |  29.42  |  21.76  |  12.78  |         19.39          | [config](charades_i3d_rgb.py) | [model](https://drive.google.com/file/d/1EFCNke077m4JC_6OMJZXEnaKW2UAgFpA/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1CuwGJ9m2YtvnKHsq9slkgtANoNEmbqVP/view?usp=sharing) |

**FineAction** with InternVideo classifier

|     Features      | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |                Config                |                                                                                         Download                                                                                          |
| :---------------: | :-----: | :------: | :------: | :------: | :----------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  VideoMAE_H_K700  |  29.44  |  19.46   |   5.06   |  19.32   |  [config](fineaction_videomae_h.py)  | [model](https://drive.google.com/file/d/1uNQufJMf9U6Igv6w4J70xiEVqYUKteTE/view?usp=sharing)  \| [log](https://drive.google.com/file/d/1VAQbtZuvRiTk8oFS7EIF9ilOKm1165u-/view?usp=sharing) |
| VideoMAEv2_g_K710 |  29.85  |  19.72   |   5.17   |  19.62   | [config](fineaction_videomaev2_g.py) | [model](https://drive.google.com/file/d/1o7HdsZIR-JufAGHD6cq-xRRIEX3IMlyY/view?usp=sharing)  \| [log](https://drive.google.com/file/d/1QenPC5OV9gI62wKkgbrdYyJpLSxP5awp/view?usp=sharing) |


## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train ActionFormer on ActivityNet dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/actionformer/anet_tsp.py
```

For more details, you can refer to the Training part in the [Usage](../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test ActionFormer on ActivityNet dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/actionformer/anet_tsp.py --checkpoint exps/anet/actionformer_tsp/gpu1_id0/checkpoint/epoch_14.pth
```

For more details, you can refer to the Test part in the [Usage](../../docs/en/usage.md).

## Citation

```latex
@inproceedings{zhang2022actionformer,
  title={Actionformer: Localizing moments of actions with transformers},
  author={Zhang, Chen-Lin and Wu, Jianxin and Li, Yin},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part IV},
  pages={492--510},
  year={2022},
  organization={Springer}
}
```