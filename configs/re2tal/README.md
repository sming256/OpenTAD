# Re2TAL

> [Re2TAL: Rewiring Pretrained Video Backbones for Reversible Temporal Action Localization](https://arxiv.org/abs/2211.14053)  
> Chen Zhao, Shuming Liu, Karttikeya Mangalam, Bernard Ghanem

<!-- [ALGORITHM] -->

## Abstract

Temporal action localization (TAL) requires long-form reasoning to predict actions of various durations and complex content. Given limited GPU memory, training TAL end to end (i.e., from videos to predictions) on long videos is a significant challenge. Most methods can only train on pre-extracted features without optimizing them for the localization problem, consequently limiting localization performance. In this work, to extend the potential in TAL networks, we propose a novel end-to-end method Re2TAL, which rewires pretrained video backbones for reversible TAL. Re2TAL builds a backbone with reversible modules, where the input can be recovered from the output such that the bulky intermediate activations can be cleared from memory during training. Instead of designing one single type of reversible module, we propose a network rewiring mechanism, to transform any module with a residual connection to a reversible module without changing any parameters. This provides two benefits: (1) a large variety of reversible networks are easily obtained from existing and even future model designs, and (2) the reversible models require much less training effort as they reuse the pre-trained parameters of their original non-reversible versions. Re2TAL, only using the RGB modality, reaches 37.01% average mAP on ActivityNet-v1.3, a new state-of-the-art record, and mAP 64.9% at tIoU=0.5 on THUMOS-14, outperforming all other RGB-only methods.

## Results and Models

To run the experiments, please download the K400-pretrained model weights of [Re2Swin-T](https://drive.google.com/file/d/1tTheMXRHk-BFVvHAL4Rjut14WWKrZ2uH/view?usp=sharing) / [Re2SlowFast-101](https://drive.google.com/file/d/1e6Zg8SrJ9UOWvy5AkhBzo1IHRJHJH-lH/view?usp=sharing), and put them under the path `./pretrained/`

**ActivityNet-1.3** with CUHK classifier.

|    Backbone     | GPUs  | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |                        Config                         |                                                                                          Download                                                                                          |
| :-------------: | :---: | :-----: | :------: | :------: | :------: | :---------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    Re2Swin-T    |   4   |  55.08  |  37.05   |   8.29   |  36.47   |  [config](e2e_anet_re2tal_actionformer_swin_tiny.py)  | [model](https://drive.google.com/file/d/1XSZ2AICZTIds_aRnuzleEh3jqPkzLAO4/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1__vYHlCBeJ6SvPRWhEbK9LYj_7kZXk9N/view?usp=sharing) |
| Re2SlowFast-101 |   4   |  55.81  |  38.49   |   9.36   |  37.55   | [config](e2e_anet_re2tal_actionformer_slowfast101.py) | [model](https://drive.google.com/file/d/1yuo2t1sgXlRD7wqFZxwntlrpe9pv6oqz/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1TnpqjP5VBre0NyxTWC40tJLrlYChfIDO/view?usp=sharing) |

**THUMOS-14**

|    Backbone     | GPUs  | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |                         Config                          |                                                                                          Download                                                                                          |
| :-------------: | :---: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-----------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    Re2Swin-T    |   2   |  82.12  |  78.09  |  70.06  |  59.81  |  44.35  |  66.89   |  [config](e2e_thumos_re2tal_actionformer_swin_tiny.py)  | [model](https://drive.google.com/file/d/1zcATsfhR2GSNGz_91ASJLR7vr0j227Th/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1gAVTN47EryzmJtycDP-8CUP9rP0F3qx_/view?usp=sharing) |
| Re2SlowFast-101 |   2   |  84.16  |  80.26  |  74.27  |  62.66  |  49.60  |  70.19   | [config](e2e_thumos_re2tal_actionformer_slowfast101.py) | [model](https://drive.google.com/file/d/1k1VHvKx72flKNtaCYp-Ttu39ylAttPT3/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1Om5ILgKayOTbLf8PjB_uYfxgdGVW59tn/view?usp=sharing) |


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