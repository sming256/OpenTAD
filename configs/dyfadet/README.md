# DyFADet

> [DyFADet: Dynamic Feature Aggregation for Temporal Action Detection](https://arxiv.org/abs/2407.03197)  
> Le Yang, Ziwei Zheng, Yizeng Han, Hao Cheng, Shiji Song, Gao Huang, Fan Li

<!-- [ALGORITHM] -->

## Abstract

Recent proposed neural network-based Temporal Action Detection (TAD) models are inherently limited to extracting the discriminative representations and modeling action instances with various lengths from complex scenes by shared-weights detection heads. Inspired by the successes in dynamic neural networks, in this paper, we build a novel dynamic feature aggregation (DFA) module that can simultaneously adapt kernel weights and receptive fields at different timestamps. Based on DFA, the proposed dynamic encoder layer aggregates the temporal features within the action time ranges and guarantees the discriminability of the extracted representations. Moreover, using DFA helps to develop a Dynamic TAD head (DyHead), which adaptively aggregates the multi-scale features with adjusted parameters and learned receptive fields better to detect the action instances with diverse ranges from videos. With the proposed encoder layer and DyHead, a new dynamic TAD model, DyFADet, achieves promising performance on a series of challenging TAD benchmarks, including HACS-Segment, THUMOS14, ActivityNet-1.3, Epic-Kitchen 100, Ego4D-Moment QueriesV1.0, and FineAction.

## Results and Models

**ActivityNet-1.3**

| Features |  Classifier  | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |        Config         |                                                                                          Download                                                                                          |
| :------: | :----------: | :-----: | :------: | :------: | :------: | :-------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   TSP    | InternVideo1 |  58.19  |  39.30   |   8.63   |  38.62   | [config](anet_tsp.py) | [model](https://drive.google.com/file/d/1RRiHR6mSJahlohINtl594VHQMuHiml34/view?usp=sharing)   \| [log](https://drive.google.com/file/d/16u9eIzhjiaBoRAgJ103PMJZDBy835QYc/view?usp=sharing) |


**THUMOS-14**

|   Features   | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |              Config              |                                                                                          Download                                                                                          |
| :----------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| VideoMAEv2-g |  85.99  |  81.66  |  76.32  |  64.46  |  50.08  |  71.70   | [config](thumos_videomaev2_g.py) | [model](https://drive.google.com/file/d/169QGt_tEEeiDw9dwHnV7TvB11-Rh6TL4/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1eDf4UMuGvCePrWGUoVX2IyGb9UEIMXRU/view?usp=sharing) |


**FineAction**

|     Features      |  Classifier  | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |                Config                |                                                                                          Download                                                                                          |
| :---------------: | :----------: | :-----: | :------: | :------: | :------: | :----------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| VideoMAEv2_g_K710 | InternVideo1 |  37.06  |  23.49   |   5.92   |  23.70   | [config](fineaction_videomaev2_g.py) | [model](https://drive.google.com/file/d/1ccbYyYGD8-BpELeBxZIV495kjKJeNm2W/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1hs9RgCdFnZuP1XXSgiNBmiIpyjXc2Rzp/view?usp=sharing) |

**HACS**

| Features | Classifier | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |           Config           |                                                                                          Download                                                                                          |
| :------: | :--------: | :-----: | :------: | :------: | :------: | :------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| SlowFast |   TCANet   |  58.09  |  40.03   |  11.96   |  39.45   | [config](hacs_slowfast.py) | [model](https://drive.google.com/file/d/1ts_oiAj5tD4e1CR4Fwr-JGcCVLbCurVy/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1OwpyP8qi31p_NCS1HRlAYwvdt6ytMVgF/view?usp=sharing) |


## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train DyFADet on THUMOS dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/dyfadet/thumos_videomaev2_g.py
```

For more details, you can refer to the Training part in the [Usage](../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test DyFADet on THUMOS dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/dyfadet/thumos_videomaev2_g.py --checkpoint exps/thumos/dyfadet_videomaev2_g/gpu1_id0/checkpoint/epoch_37.pth
```

For more details, you can refer to the Test part in the [Usage](../../docs/en/usage.md).

## Citation

```latex
@inproceedings{yang2024dyfadet,
  title={DyFADet: Dynamic Feature Aggregation for Temporal Action Detection},
  author={Yang, Le and Zheng, Ziwei and Han, Yizeng and Cheng, Hao and Song, Shiji and Huang, Gao and Li, Fan},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```