# TriDet

> [TriDet: Temporal Action Detection with Relative Boundary Modeling](https://arxiv.org/abs/2303.07347)  
> Dingfeng Shi, Yujie Zhong, Qiong Cao, Lin Ma, Jia Li, Dacheng Tao

<!-- [ALGORITHM] -->

## Abstract

In this paper, we present a one-stage framework TriDet for temporal action detection. Existing methods often suffer from imprecise boundary predictions due to the ambiguous action boundaries in videos. To alleviate this problem, we propose a novel Trident-head to model the action boundary via an estimated relative probability distribution around the boundary. In the feature pyramid of TriDet, we propose an efficient Scalable-Granularity Perception (SGP) layer to mitigate the rank loss problem of self-attention that takes place in the video features and aggregate information across different temporal granularities. Benefiting from the Trident-head and the SGP-based feature pyramid, TriDet achieves state-of-the-art performance on three challenging benchmarks: THUMOS14, HACS and EPIC-KITCHEN 100, with lower computational costs, compared to previous methods. For example, TriDet hits an average mAP of 69.3% on THUMOS14, outperforming the previous best by 2.5%, but with only 74.6% of its latency. 

## Results and Models

**ActivityNet-1.3** with CUHK classifier.

| Features | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |        Config         |                                                                                          Download                                                                                          |
| :------: | :-----: | :------: | :------: | :------: | :-------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   TSP    |  54.89  |  38.20   |   8.21   |  36.96   | [config](anet_tsp.py) | [model](https://drive.google.com/file/d/16MmZlFDBMjr9TK-nit6eKCksQYEASwHZ/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1WY6x9OeXDvaWRdHVc9-R8e5lUjWim020/view?usp=sharing) |

**THUMOS-14**

| Features | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |         Config          |                                                                                          Download                                                                                          |
| :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :---------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   I3D    |  84.46  |  81.05  |  73.41  |  62.58  |  46.51  |  69.60   | [config](thumos_i3d.py) | [model](https://drive.google.com/file/d/18nQ2jJMJkkqeCfWzj-OfV0Hi_Ej3Y1lN/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1YLXoZoliAD1jLHa2Lxtct0EFtpgabesM/view?usp=sharing) |

**HACS**

| Features | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |           Config           |                                                                                          Download                                                                                          |
| :------: | :-----: | :------: | :------: | :------: | :------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| SlowFast |  56.84  |  39.04   |  11.13   |  38.47   | [config](hacs_slowfast.py) | [model](https://drive.google.com/file/d/13e34x-kXRWTm9jNkvv61D10vRdBtRgZ_/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1RGwwtenzxiZ5IBw7SdF9V2bFNk5TBZq1/view?usp=sharing) |

**Epic-Kitchens-100**

| Subset | Features | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | ave. mAP |                  Config                  |                                                                                          Download                                                                                          |
| :----: | :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :--------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  Noun  | SlowFast |  24.95  |  23.76  |  22.22  |  20.00  |  16.63  |  21.51   | [config](epic_kitchens_slowfast_noun.py) | [model](https://drive.google.com/file/d/18F0Ruc9RJL9KLOjIEFvBGBQzXxNUkJgH/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1Tfi4KyKmnanO2F3rry1hIRDtlvKZDRTT/view?usp=sharing) |
|  Verb  | SlowFast |  27.88  |  27.00  |  25.52  |  23.74  |  20.72  |  24.97   | [config](epic_kitchens_slowfast_verb.py) | [model](https://drive.google.com/file/d/1LIGCduQFPXKQl98s53h_feIYNLBsXkXd/view?usp=sharing)   \| [log](https://drive.google.com/file/d/10xZSt7r6T_tauqvc9UHi3bCiloZPATDp/view?usp=sharing) |


## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TriDet on THUMOS dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/tridet/thumos_i3d.py
```

For more details, you can refer to the Training part in the [Usage](../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TriDet on THUMOS dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/tridet/thumos_i3d.py --checkpoint exps/thumos/tridet_i3d/gpu1_id0/checkpoint/epoch_37.pth
```

For more details, you can refer to the Test part in the [Usage](../../docs/en/usage.md).


## Citation

```latex
@inproceedings{shi2023tridet,
  title={TriDet: Temporal Action Detection with Relative Boundary Modeling},
  author={Shi, Dingfeng and Zhong, Yujie and Cao, Qiong and Ma, Lin and Li, Jia and Tao, Dacheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18857--18866},
  year={2023}
}
```