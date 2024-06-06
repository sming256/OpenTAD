# ETAD

> [ETAD: Training Action Detection End to End on a Laptop](https://arxiv.org/abs/2205.07134)  
> Shuming Liu, Mengmeng Xu, Chen Zhao, Xu Zhao, Bernard Ghanem

<!-- [ALGORITHM] -->

## Abstract

Temporal action detection (TAD) with end-to-end training often suffers from the pain of huge demand for computing resources due to long video duration. In this work, we propose an efficient temporal action detector (ETAD) that can train directly from video frames with extremely low GPU memory consumption. Our main idea is to minimize and balance the heavy computation among features and gradients in each training iteration. We propose to sequentially forward the snippet frame through the video encoder, and backward only a small necessary portion of gradients to update the encoder. To further alleviate the computational redundancy in training, we propose to dynamically sample only a small subset of proposals during training. Moreover, various sampling strategies and ratios are studied for both the encoder and detector. ETAD achieves state-of-the-art performance on TAD benchmarks with remarkable efficiency. On ActivityNet-1.3, training ETAD in 18 hours can reach 38.25% average mAP with only 1.3 GB memory consumption per video under end-to-end training. Our code will be publicly released.

## Results and Models

**ActivityNet-1.3** with CUHK classifier.

|  E2E  | Features | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |          Config           |                                                                                          Download                                                                                          |
| :---: | :------: | :-----: | :------: | :------: | :------: | :-----------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| False |   TSP    |  54.91  |  38.90   |   9.09   |  37.73   |   [config](anet_tsp.py)   | [model](https://drive.google.com/file/d/1r7zLfIaLPHb_7ba6tPzjKE3_WpGtrnge/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1wc27fX4qWUBarzeXR2aDfQjvN25G72Uu/view?usp=sharing) |
| True  |   TSP    |  55.82  |  40.12   |  10.30   |  38.76   | [config](e2e_anet_tsp.py) | [model](https://drive.google.com/file/d/1SJObHl-vB_0MSJCzRLWN_xJxF2aUnhSu/view?usp=sharing)   \| [log](https://drive.google.com/file/d/16t8A3doleSsTtZeJ4rc-hUML9_8gltT_/view?usp=sharing) |
- To run the end-to-end experiments, please download the converted TSP-R(2+1d) model (ActivityNet pretrained) from [here](https://drive.google.com/file/d/1vFnMwl3tAiPtmYFWzHxP07q1AQsCzPPN/view?usp=sharing) and put it in the `./pretrained/` folder.


**THUMOS-14** with UtrimmedNet classifier

|  E2E  | Features | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |         Config          |                                                                                          Download                                                                                          |
| :---: | :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :---------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| False |   I3D    |  67.74  |  64.22  |  58.23  |  49.19  |  38.41  |  55.56   | [config](thumos_i3d.py) | [model](https://drive.google.com/file/d/1sRPYxfD_jGmsa1jB4Kw2Z4bVVnnlxx3u/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1T4QHGOz5BZxHsH_vt8_u6aFYKHoZZeuY/view?usp=sharing) |


## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train ETAD on ActivityNet dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/etad/anet_tsp.py
```

For more details, you can refer to the Training part in the [Usage](../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test ETAD on ActivityNet dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/etad/anet_tsp.py --checkpoint exps/anet/etad_tsp_128/gpu1_id0/checkpoint/epoch_5.pth
```

For more details, you can refer to the Test part in the [Usage](../../docs/en/usage.md).

## Citation

```latex
@inproceedings{liu2023etad,
  title={ETAD: Training Action Detection End to End on a Laptop},
  author={Liu, Shuming and Xu, Mengmeng and Zhao, Chen and Zhao, Xu and Ghanem, Bernard},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4524--4533},
  year={2023}
}
```