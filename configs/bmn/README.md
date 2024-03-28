# BMN

> [BMN: Boundary-Matching Network for Temporal Action Proposal Generation](https://arxiv.org/abs/1907.09702)  
> Tianwei Lin, Xiao Liu, Xin Li, Errui Ding, Shilei Wen

<!-- [ALGORITHM] -->

## Abstract

Temporal action proposal generation is an challenging and promising task which aims to locate temporal regions in real-world videos where action or event may occur. Current bottom-up proposal generation methods can generate proposals with precise boundary, but cannot efficiently generate adequately reliable confidence scores for retrieving proposals. To address these difficulties, we introduce the Boundary-Matching (BM) mechanism to evaluate confidence scores of densely distributed proposals, which denote a proposal as a matching pair of starting and ending boundaries and combine all densely distributed BM pairs into the BM confidence map. Based on BM mechanism, we propose an effective, efficient and end-to-end proposal generation method, named Boundary-Matching Network (BMN), which generates proposals with precise temporal boundaries as well as reliable confidence scores simultaneously. The two-branches of BMN are jointly trained in an unified framework. We conduct experiments on two challenging datasets: THUMOS-14 and ActivityNet-1.3, where BMN shows significant performance improvement with remarkable efficiency and generalizability. Further, combining with existing action classifier, BMN can achieve state-of-the-art temporal action detection performance.

## Results and Models

**ActivityNet-1.3** with CUHK classifier.

| Features | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |        Config         |                                                                                          Download                                                                                          |
| :------: | :-----: | :------: | :------: | :------: | :-------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   TSN    |  50.97  |  34.98   |   8.35   |  34.21   | [config](anet_tsn.py) | [model](https://drive.google.com/file/d/1yWu05_BS2RuHnItIpFvKVieiYOGWbzNj/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1KtICNUOt_IVEWfSXqZ-6R_msKm40AaTU/view?usp=sharing) |
|   TSP    |  52.90  |  37.30   |   9.67   |  36.40   | [config](anet_tsp.py) | [model](https://drive.google.com/file/d/1sQQble7270Ja5Nx8kR5tApeJGGS42t6M/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1sxXjeIzEt0Nx_YZNj_tCcAsN9_TJe6xU/view?usp=sharing) |

Use above checkpoints to evaluate the recall performance:

| Features | AR@1  | AR@5  | AR@10 | AR@100 |  AUC  |            Config            |                                                                                         Download                                                                                          |
| :------: | :---: | :---: | :---: | :----: | :---: | :--------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   TSN    | 33.58 | 49.16 | 56.53 | 75.34  | 67.23 | [config](anet_tsn_recall.py) | [model](https://drive.google.com/file/d/1yWu05_BS2RuHnItIpFvKVieiYOGWbzNj/view?usp=sharing)  \| [log](https://drive.google.com/file/d/1xT6T7s8tNcuVgItWXyF2gI2cCU6T993O/view?usp=sharing) |
|   TSP    | 34.14 | 51.35 | 58.44 | 76.24  | 68.47 | [config](anet_tsp_recall.py) | [model](https://drive.google.com/file/d/1sQQble7270Ja5Nx8kR5tApeJGGS42t6M/view?usp=sharing)  \| [log](https://drive.google.com/file/d/12Q5vhXaH_YQoAxqtWFAzePAb5MfGGJ1Z/view?usp=sharing) |


**THUMOS-14** with UtrimmedNet classifier.

| Features | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |         Config          |                                                                                          Download                                                                                          |
| :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :---------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   TSN    |  60.51  |  56.03  |  47.56  |  38.23  |  28.64  |  46.19   | [config](thumos_tsn.py) | [model](https://drive.google.com/file/d/14GANupLwRE4dAesyafeU4aGx-8c38ogd/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1DeDCLezP4rRZwq7ppPG5coA2ol0fSvkZ/view?usp=sharing) |
|   I3D    |  64.99  |  60.70  |  54.54  |  44.11  |  34.16  |  51.70   | [config](thumos_i3d.py) | [model](https://drive.google.com/file/d/1aPl_jNOlbcr7Qc2DzNM10dRiMDfcSJ7V/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1TJGylck9Zhqew5IECeDjqwLNO8Zkq0QD/view?usp=sharing) |

**HACS** with TCANet classifier.

| Features | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |             Config             |                                                                                          Download                                                                                          |
| :------: | :-----: | :------: | :------: | :------: | :----------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| SlowFast |  52.64  |  36.18   |  11.46   |  35.78   | [config](hacs_slowfast_192.py) | [model](https://drive.google.com/file/d/1gLOmyu-zLi7Gpph2czH9bgiJCKyeUvsm/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1fMm1sjNtbgWNIcFmdRSpnvKfiXOmsLQ2/view?usp=sharing) |


## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train BMN on ActivityNet dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/bmn/anet_tsp.py
```

For more details, you can refer to the Training part in the [Usage](../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test BMN on ActivityNet dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/bmn/anet_tsp.py --checkpoint exps/anet/bmn_tsp_128/gpu1_id0/checkpoint/epoch_9.pth
```

To test the recal performance:

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/bmn/anet_tsp_recall.py --checkpoint exps/anet/bmn_tsp_128/gpu1_id0/checkpoint/epoch_9.pth
```

For more details, you can refer to the Test part in the [Usage](../../docs/en/usage.md).


## Citation

```latex
@inproceedings{lin2019bmn,
  title={Bmn: Boundary-matching network for temporal action proposal generation},
  author={Lin, Tianwei and Liu, Xiao and Li, Xin and Ding, Errui and Wen, Shilei},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={3889--3898},
  year={2019}
}
```