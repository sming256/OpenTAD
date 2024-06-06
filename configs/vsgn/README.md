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
|   False   | Fasle |   TSP    |  54.40  |  36.88   |   9.15   |  36.27   |   [config](anet_tsp.py)   | [model](https://drive.google.com/file/d/1E85NhJyyO_6qfnucfVg5mEyW9t6UJH7v/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1g1UVUtd-6UnNVawist6cvHaGEsnlZHxP/view?usp=sharing) |
|   False   | True  |   TSP    |  54.80  |  37.35   |   9.80   |  36.89   | [config](anet_tsp_gcn.py) | [model](https://drive.google.com/file/d/1y7PqP1TiNkGNxx-WLBccZvuTUo2mfbOD/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1krOBqbYNI9moRbIhbYpOT6RUphXIQDD9/view?usp=sharing) |

**THUMOS-14**

| Stitching |  GCN  | Features | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |           Config            |                                                                                          Download                                                                                          |
| :-------: | :---: | :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   False   | False |   TSN    |  59.14  |  53.18  |  45.49  |  35.11  |  24.69  |  43.52   |   [config](thumos_tsn.py)   | [model](https://drive.google.com/file/d/1zNW8TDTA1C3BDR1Fu81XFg-q0iqxneKR/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1X1IEsQq4IEGTDt_ISJpY9elxqxiRZTT9/view?usp=sharing) |
|   False   | True  |   TSN    |  61.75  |  56.69  |  47.62  |  37.16  |  26.70  |  45.98   | [config](thumos_tsn_gcn.py) | [model](https://drive.google.com/file/d/1itc8CuYhuk4Ja8T_rl4vATqbI-xOVZPz/view?usp=sharing)   \| [log](https://drive.google.com/file/d/10Q_B7k_fQ-7U1_xvnbzP0u1jhnuSypMK/view?usp=sharing) |
|   False   | False |   I3D    |  66.78  |  61.68  |  53.09  |  43.08  |  30.63  |  51.05   |   [config](thumos_i3d.py)   | [model](https://drive.google.com/file/d/1qW2QvF9WIpbPZtySqoMwpdRXnAIiZKew/view?usp=sharing)   \| [log](https://drive.google.com/file/d/18kjVV-dcU0sAsrI99g7gpo5n-eyUqbgk/view?usp=sharing) |
|   False   | True  |   I3D    |  68.64  |  63.39  |  55.21  |  44.74  |  33.13  |  53.02   | [config](thumos_i3d_gcn.py) | [model](https://drive.google.com/file/d/1TvXF_jc4KHzlkwaRKFNZ0NuzXz7EioMo/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1nm0-y0c7_RS4MyVZbhrjtbS_MAUjhOx6/view?usp=sharing) |


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