# Video Mamba Suite

> [Video Mamba Suite: State Space Model as a Versatile Alternative for Video Understanding](https://arxiv.org/abs/2403.09626)  
> Guo Chen, Yifei Huang, Jilan Xu, Baoqi Pei, Zhe Chen, Zhiqi Li, Jiahao Wang, Kunchang Li, Tong Lu, Limin Wang

<!-- [ALGORITHM] -->

## Abstract

Understanding videos is one of the fundamental directions in computer vision research, with extensive efforts dedicated to exploring various architectures such as RNN, 3D CNN, and Transformers. The newly proposed architecture of state space model, e.g., Mamba, shows promising traits to extend its success in long sequence modeling to video modeling. To assess whether Mamba can be a viable alternative to Transformers in the video understanding domain, in this work, we conduct a comprehensive set of studies, probing different roles Mamba can play in modeling videos, while investigating diverse tasks where Mamba could exhibit superiority. We categorize Mamba into four roles for modeling videos, deriving a Video Mamba Suite composed of 14 models/modules, and evaluating them on 12 video understanding tasks. Our extensive experiments reveal the strong potential of Mamba on both video-only and video-language tasks while showing promising efficiency-performance trade-offs. We hope this work could provide valuable data points and insights for future research on video understanding.

## Usage

Before running the TAD experiments, go to [video-mamba-suite official repo](https://github.com/OpenGVLab/video-mamba-suite?tab=readme-ov-file#preliminary-installation) and install the mamba module as the library.


## Results and Models

**ActivityNet-1.3** with InternVideo2 classifier.

|    Features     | Setting | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |             Config              |                                                                                          Download                                                                                          |
| :-------------: | :-----: | :-----: | :------: | :------: | :------: | :-----------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| InternVideo2-6B |   DBM   |  63.13  |  44.36   |  10.36   |  42.80   | [config](anet_internvideo6b.py) | [model](https://drive.google.com/file/d/1TdPE40iYlAp8f66Cnlb_Mbc9OBt-ajzQ/view?usp=sharing)   \| [log](https://drive.google.com/file/d/13B8BwtIHEKjnKlJDvyELfr71izYsI-mn/view?usp=sharing) |

 - The validation dataset we used has 4,728 videos, which is the same number as in BMN but less than ActionFormer's implementation. Consequently, this result is slightly higher than VideoMambaSuite's official result. You can check [README](../../tools/prepare_data/activitynet/README.md) for more details.

**THUMOS-14**

|    Features     | Setting | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |              Config               |                                                                                          Download                                                                                          |
| :-------------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| InternVideo2-6B |   DBM   |  87.30  |  82.95  |  77.17  |  67.06  |  51.74  |  73.24   | [config](thumos_internvideo6b.py) | [model](https://drive.google.com/file/d/1l51lWq3ljmxdV3Tal95JMY7yZv4tU_70/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1yPh545NkTHKppbHQ7Ul2vUHuMnA9MRFF/view?usp=sharing) |

- Following VSGN, we additionally delete `video_test_0000270` during testing due to wrong annotation. Consequently, this result is slightly higher than VideoMambaSuite's official result. You can check [README](../../tools/prepare_data/thumos/README.md) for more details.

**FineAction**

|    Features     | Setting | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |                Config                 |                                                                                          Download                                                                                          |
| :-------------: | :-----: | :-----: | :------: | :------: | :------: | :-----------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| InternVideo2-1B |   DBM   |  45.72  |  29.37   |   5.37   |  29.13   | [config](fineaction_internvideo1b.py) | [model](https://drive.google.com/file/d/12nfbV0xG5EtM5XLk6nO0L2lGtJy4udKi/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1NIR-qfHGuuAH3bD_wkFBqTw-59msDpJ6/view?usp=sharing) |

**HACS** with InternVideo2 classifier.

|    Features     | Setting | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |             Config              |                                                                                          Download                                                                                          |
| :-------------: | :-----: | :-----: | :------: | :------: | :------: | :-----------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| InternVideo2-6B |   DBM   |  64.16  |  46.08   |  13.86   |  44.81   | [config](hacs_internvideo6b.py) | [model](https://drive.google.com/file/d/1qRAYLEBDn6bI4SeGIf3vU2gkf3d-w32P/view?usp=sharing)   \| [log](https://drive.google.com/file/d/19P6ISpbiXlTz2fggeI1rSGivjhRVC6fG/view?usp=sharing) |

**MultiTHUMOS**

|    Features     | Setting | mAP@0.2 | mAP@0.5 | mAP@0.7 | ave. mAP (0.1:0.9:0.1) |                 Config                 |                                                                                          Download                                                                                          |
| :-------------: | :-----: | :-----: | :-----: | :-----: | :--------------------: | :------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| InternVideo2-6B |   DBM   |  65.71  |  51.57  |  31.09  |         44.58          | [config](multithumos_internvideo6b.py) | [model](https://drive.google.com/file/d/1u39RJ1T69IsVzV4H2EQJebgsVx_KP1yh/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1VNRT0bYpjYm3KKEMmqaGq2AsGdozvSyi/view?usp=sharing) |


## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train VideoMambaSuite on ActivityNet dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/videomambasuite/videomambasuite_internvideo6b.py
```

For more details, you can refer to the Training part in the [Usage](../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test VideoMambaSuite on ActivityNet dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/videomambasuite/videomambasuite_internvideo6b.py --checkpoint exps/anet/videomambasuite_internvideo6b/gpu1_id0/checkpoint/epoch_11.pth
```

For more details, you can refer to the Test part in the [Usage](../../docs/en/usage.md).

## Citation

```latex
@misc{2024videomambasuite,
      title={Video Mamba Suite: State Space Model as a Versatile Alternative for Video Understanding}, 
      author={Guo Chen, Yifei Huang, Jilan Xu, Baoqi Pei, Zhe Chen, Zhiqi Li, Jiahao Wang, Kunchang Li, Tong Lu, Limin Wang},
      year={2024},
      eprint={2403.09626},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```