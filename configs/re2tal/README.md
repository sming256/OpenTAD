# Re2TAL

> [Re2TAL: Rewiring Pretrained Video Backbones for Reversible Temporal Action Localization](https://arxiv.org/abs/2211.14053)  
> Chen Zhao, Shuming Liu, Karttikeya Mangalam, Bernard Ghanem

<!-- [ALGORITHM] -->

## Abstract

Temporal action localization (TAL) requires long-form reasoning to predict actions of various durations and complex content. Given limited GPU memory, training TAL end to end (i.e., from videos to predictions) on long videos is a significant challenge. Most methods can only train on pre-extracted features without optimizing them for the localization problem, consequently limiting localization performance. In this work, to extend the potential in TAL networks, we propose a novel end-to-end method Re2TAL, which rewires pretrained video backbones for reversible TAL. Re2TAL builds a backbone with reversible modules, where the input can be recovered from the output such that the bulky intermediate activations can be cleared from memory during training. Instead of designing one single type of reversible module, we propose a network rewiring mechanism, to transform any module with a residual connection to a reversible module without changing any parameters. This provides two benefits: (1) a large variety of reversible networks are easily obtained from existing and even future model designs, and (2) the reversible models require much less training effort as they reuse the pre-trained parameters of their original non-reversible versions. Re2TAL, only using the RGB modality, reaches 37.01% average mAP on ActivityNet-v1.3, a new state-of-the-art record, and mAP 64.9% at tIoU=0.5 on THUMOS-14, outperforming all other RGB-only methods.

## Results and Models

To run the experiments, please download the K400-pretrained model weights of [Re2Swin-T](https://drive.google.com/file/d/1qWplk_O1PW5oCLRckfrPgwlT4nZDC49Y/view?usp=sharing) / [Re2SlowFast-101](https://drive.google.com/file/d/1FJ_7dxYPobheqCdkRKOZ9KaGNTNp6zx1/view?usp=sharing), and put them under the path `./pretrained/`

**ActivityNet-1.3** with CUHK classifier.

|    Backbone     | GPUs  | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |                        Config                         |                                                                                          Download                                                                                          |
| :-------------: | :---: | :-----: | :------: | :------: | :------: | :---------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|     Swin-T      |   4   |  55.34  |  37.68   |   9.12   |  36.89   |     [config](e2e_anet_actionformer_swin_tiny.py)      | [model](https://drive.google.com/file/d/1TnzgBUKJKL1swL46WMqGNwS9BL2NMmBP/view?usp=sharing)   \| [log](https://drive.google.com/file/d/13-u4-Sv66sW29sUAsYInFulb0tBwxpvf/view?usp=sharing) |
|  SlowFast-101   |   4   |  55.48  |  38.32   |   9.34   |  37.35   |    [config](e2e_anet_actionformer_slowfast101.py)     | [model](https://drive.google.com/file/d/1BDriRwR1pWqwcX72aCL6qx-gpBtEXSZq/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1lMoiWAfSiWYXiuiam6kUIJOyY35qV_ip/view?usp=sharing) |
|    Re2Swin-T    |   4   |  55.08  |  37.05   |   8.29   |  36.47   |  [config](e2e_anet_re2tal_actionformer_swin_tiny.py)  | [model](https://drive.google.com/file/d/19s4xELNaLnuZuQqVnAreC5hbRgFvXwpU/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1JN2qgKHmYDVj0bTGv855JEJ_YgNr1YgG/view?usp=sharing) |
| Re2SlowFast-101 |   4   |  55.81  |  38.49   |   9.36   |  37.55   | [config](e2e_anet_re2tal_actionformer_slowfast101.py) | [model](https://drive.google.com/file/d/180djXTqt2uE2Q5e78avp080tOAdoSzcb/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1S6mulU6TTxfyIBQz30CCd_P7JtUJs8VE/view?usp=sharing) |

- We use activation checkpointing to save the memory in non-reversible models.

**THUMOS-14**

|    Backbone     | GPUs  | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |                         Config                          |                                                                                          Download                                                                                          |
| :-------------: | :---: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-----------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|     Swin-T      |   2   |  82.83  |  78.70  |  71.33  |  58.79  |  43.85  |  67.10   |     [config](e2e_thumos_actionformer_swin_tiny.py)      | [model](https://drive.google.com/file/d/1lFEFYjg0AWzrOoKjGGLTLb-3igelCys4/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1h2DtNsYylPSl4D0ZGHsQ5Z_wDjmnk4s-/view?usp=sharing) |
|  SlowFast-101   |   2   |  83.99  |  80.38  |  75.29  |  62.13  |  48.08  |  69.97   |    [config](e2e_thumos_actionformer_slowfast101.py)     | [model](https://drive.google.com/file/d/1d2U47XIkNtwSiq3b5rTX7v4UEAPO-GqG/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1ZLZuse_rmMb_WzDokFnLrWc3aYB_IbOg/view?usp=sharing) |
|    Re2Swin-T    |   2   |  82.12  |  78.09  |  70.06  |  59.81  |  44.35  |  66.89   |  [config](e2e_thumos_re2tal_actionformer_swin_tiny.py)  | [model](https://drive.google.com/file/d/1zqVtgjPc9FB8K_JMdnK8s0cp1yeKy2WY/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1iBMuX8zRQkhnH7vQZcoj00lHEgCjrTO9/view?usp=sharing) |
| Re2SlowFast-101 |   2   |  84.16  |  80.26  |  74.27  |  62.66  |  49.60  |  70.19   | [config](e2e_thumos_re2tal_actionformer_slowfast101.py) | [model](https://drive.google.com/file/d/1Zgk708kvjX_iSNlzGmSOjBX62A3qhNar/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1kX_WtnLSH7m1GrWkWlipefJOTuKsiVd8/view?usp=sharing) |

- We use activation checkpointing to save the memory in non-reversible models.

## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train Re2TAL on ActivityNet dataset with **4** GPUs.

```shell
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/re2tal/e2e_anet_re2tal_actionformer_slowfast101.py
```

For more details, you can refer to the Training part in the [Usage](../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test Re2TAL on ActivityNet dataset with **4** GPUs.

```shell
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/re2tal/e2e_anet_re2tal_actionformer_slowfast101.py --checkpoint exps/anet/re2tal_e2e_actionformer_slowfast101_frame768/gpu4_id0/checkpoint/epoch_14.pth
```

For more details, you can refer to the Test part in the [Usage](../../docs/en/usage.md).


## Citation

```latex
@inproceedings{zhao2023re2tal,
  title={Re2tal: Rewiring pretrained video backbones for reversible temporal action localization},
  author={Zhao, Chen and Liu, Shuming and Mangalam, Karttikeya and Ghanem, Bernard},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10637--10647},
  year={2023}
}
```