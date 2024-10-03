# VSGN

> [Video Self-Stitching Graph Network for Temporal Action Localization](https://arxiv.org/abs/2011.14598)  
> Chen Zhao, Ali Thabet, Bernard Ghanem

<!-- [ALGORITHM] -->

## Abstract

Temporal action localization (TAL) in videos is a challenging task, especially due to the large variation in action temporal scales. Short actions usually occupy the major proportion in the data, but have the lowest performance with all current methods. In this paper, we confront the challenge of short actions and propose a multi-level cross-scale solution dubbed as video self-stitching graph network (VSGN). We have two key components in VSGN: video self-stitching (VSS) and cross-scale graph pyramid network (xGPN). In VSS, we focus on a short period of a video and magnify it along the temporal dimension to obtain a larger scale. We stitch the original clip and its magnified counterpart in one input sequence to take advantage of the complementary properties of both scales. The xGPN component further exploits the cross-scale correlations by a pyramid of cross-scale graph networks, each containing a hybrid module to aggregate features from across scales as well as within the same scale. Our VSGN not only enhances the feature representations, but also generates more positive anchors for short actions and more short training samples. Experiments demonstrate that VSGN obviously improves the localization performance of short actions as well as achieving the state-of-the-art overall performance on THUMOS-14 and ActivityNet-v1.3.

## Results and Models

**ActivityNet-1.3** with CUHK classifier.

| Stitching |  GCN  | Features | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |          Config           |                                                                                          Download                                                                                          |
| :-------: | :---: | :------: | :-----: | :------: | :------: | :------: | :-----------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   False   | Fasle |   TSP    |  54.40  |  36.88   |   9.15   |  36.27   |   [config](anet_tsp.py)   | [model](https://drive.google.com/file/d/15bbt9hcE0LwLKONqv3LKEa6-8QEqNgwX/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1HuoO75OIZgjoNZLfkj-Cl49-vjLD6QWy/view?usp=sharing) |
|   False   | True  |   TSP    |  54.80  |  37.35   |   9.80   |  36.89   | [config](anet_tsp_gcn.py) | [model](https://drive.google.com/file/d/1TIHqpfmbAFMNhVGAU-YKD0ZS3cd-VOh9/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1LW1A_gIpOqz1RDiVgQDYx3BT-NG_-Uzj/view?usp=sharing) |

**THUMOS-14**

| Stitching |  GCN  | Features | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |           Config            |                                                                                          Download                                                                                          |
| :-------: | :---: | :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   False   | False |   TSN    |  59.14  |  53.18  |  45.49  |  35.11  |  24.69  |  43.52   |   [config](thumos_tsn.py)   | [model](https://drive.google.com/file/d/1JtEvIOGabOod3ZzKpHEpdyu-FRoj4ASQ/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1YeOz4z4KaZzFbKiuIHm6OXy8hdyf2XNJ/view?usp=sharing) |
|   False   | True  |   TSN    |  61.75  |  56.69  |  47.62  |  37.16  |  26.70  |  45.98   | [config](thumos_tsn_gcn.py) | [model](https://drive.google.com/file/d/1TA54ay5HHadW7R3xjplNvFRZRGIBGfFi/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1xeOVqFlX_oeFzxcCLxC_qMnHeNoawjaO/view?usp=sharing) |
|   False   | False |   I3D    |  66.78  |  61.68  |  53.09  |  43.08  |  30.63  |  51.05   |   [config](thumos_i3d.py)   | [model](https://drive.google.com/file/d/1cw3hViPLKWXmHrENy5rpeOQhoLcxnWDG/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1kZaoTuO2RdJJu1Hetj65f_wzuTqY0fi-/view?usp=sharing) |
|   False   | True  |   I3D    |  68.64  |  63.39  |  55.21  |  44.74  |  33.13  |  53.02   | [config](thumos_i3d_gcn.py) | [model](https://drive.google.com/file/d/1bthK_X0M8x9yE8Yj2O-S9fLuVgyJznVT/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1a65Q46H6B-qlEn_eWp-qIYYgMsvLG1E6/view?usp=sharing) |


## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train VSGN on ActivityNet dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/vsgn/anet_tsp.py
```

For more details, you can refer to the Training part in the [Usage](../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test VSGN on ActivityNet dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/vsgn/anet_tsp.py --checkpoint exps/anet/vsgn_tsp_256/gpu1_id0/checkpoint/epoch_9.pth
```

For more details, you can refer to the Test part in the [Usage](../../docs/en/usage.md).

## Citation

```latex
@inproceedings{zhao2021video,
  title={Video self-stitching graph network for temporal action localization},
  author={Zhao, Chen and Thabet, Ali K and Ghanem, Bernard},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13658--13667},
  year={2021}
}
```