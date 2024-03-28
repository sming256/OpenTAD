# AFSD

> [Learning Salient Boundary Feature for Anchor-free Temporal Action Localization](https://arxiv.org/abs/2103.13137)  
> Chuming Lin, Chengming Xu, Donghao Luo, Yabiao Wang, Ying Tai, Chengjie Wang, Jilin Li, Feiyue Huang, Yanwei Fu

<!-- [ALGORITHM] -->

## Abstract

Temporal action localization is an important yet challenging task in video understanding. Typically, such a task aims at inferring both the action category and localization of the start and end frame for each action instance in a long, untrimmed video.While most current models achieve good results by using pre-defined anchors and numerous actionness, such methods could be bothered with both large number of outputs and heavy tuning of locations and sizes corresponding to different anchors. Instead, anchor-free methods is lighter, getting rid of redundant hyper-parameters, but gains few attention. In this paper, we propose the first purely anchor-free temporal localization method, which is both efficient and effective. Our model includes (i) an end-to-end trainable basic predictor, (ii) a saliency-based refinement module to gather more valuable boundary features for each proposal with a novel boundary pooling, and (iii) several consistency constraints to make sure our model can find the accurate boundary given arbitrary proposals. Extensive experiments show that our method beats all anchor-based and actionness-guided methods with a remarkable margin on THUMOS14, achieving state-of-the-art results, and comparable ones on ActivityNet v1.3.

## Results and Models

**ActivityNet-1.3** with CUHK classifier.

|  E2E  |      Setting      | GPUs  | mAP@0.5 | mAP@0.75 | mAP@0.95 | Average mAP |                     Config                      |                                                                                         Download                                                                                          |
| :---: | :---------------: | :---: | :-----: | :------: | :------: | :---------: | :---------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| False |    Feature-TSP    |   1   |  54.44  |  36.72   |   8.69   |    36.10    |              [config](anet_tsp.py)              | [model](https://drive.google.com/file/d/1wha6AM-FuSR9zxZAo4LiTiSNFIyXo-Em/view?usp=sharing)  \| [log](https://drive.google.com/file/d/1pW8o6YH4ek33Fkq4EpsS_i2J1EQ-y9tS/view?usp=sharing) |
| True  | I3D-R50-768x96x96 |   4   |  52.77  |  35.01   |   7.74   |    34.57    | [config](anet_i3d_r50_768x96x96_lr1e-5_bs16.py) | [model](https://drive.google.com/file/d/1lJT0nm3GHCxTt-SYLlke_nd3oufuILmo/view?usp=sharing)  \| [log](https://drive.google.com/file/d/1pER_OyZ0N5cfr2z0LglZE9Xh1PCx8plL/view?usp=sharing) |


**THUMOS14**
|  E2E  |      Setting      | GPUs  | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | Average mAP |                    Config                    |                                                                                          Download                                                                                          |
| :---: | :---------------: | :---: | :-----: | :-----: | :-----: | :-----: | :-----: | :---------: | :------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| False |    Feature-I3D    |   1   |  73.20  |  68.45  |  60.16  |  46.74  |  31.24  |    55.96    |           [config](thumos_i3d.py)            | [model](https://drive.google.com/file/d/1j_GIAY8QOeasuXiGw56WJ5jzRqQYnYw7/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1JCtl56tDJCCMVeVUaYFFTuLfp_fj3X5b/view?usp=sharing) |
| True  | I3D-R50-256x96x96 |   1   |  3.88   |  48.81  |  41.36  |  31.70  |  21.03  |    39.36    | [config](thumos_i3d_r50_256x96x96_lr1e-5.py) | [model](https://drive.google.com/file/d/1ad4l6l0fYngEkX0H3xgQaSmhkYjVcmxw/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1zhts1Uh6eqxQfxcmVa6atWY5HsORBOyM/view?usp=sharing) |


## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train AFSD on ActivityNet dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/afsd/anet_tsp.py
```

For more details, you can refer to the Training part in the [Usage](../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test AFSD on ActivityNet dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/afsd/anet_tsp.py --checkpoint exps/anet/afsd_tsp_96/gpu1_id0/checkpoint/epoch_9.pth
```

For more details, you can refer to the Test part in the [Usage](../../docs/en/usage.md).

## Citation

```latex
@InProceedings{Lin_2021_CVPR,
    author    = {Lin, Chuming and Xu, Chengming and Luo, Donghao and Wang, Yabiao and Tai, Ying and Wang, Chengjie and Li, Jilin and Huang, Feiyue and Fu, Yanwei},
    title     = {Learning Salient Boundary Feature for Anchor-free Temporal Action Localization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {3320-3329}
}
```