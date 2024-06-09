# AdaTAD

> [End-to-End Temporal Action Detection with 1B Parameters Across 1000 Frames](https://arxiv.org/abs/2311.17241)  
> Shuming Liu, Chen-Lin Zhang, Chen Zhao, Bernard Ghanem

<!-- [ALGORITHM] -->

## Abstract

Recently, temporal action detection (TAD) has seen significant performance improvement with end-to-end training. However, due to the memory bottleneck, only models with limited scales and limited data volumes can afford end-to-end training, which inevitably restricts TAD performance. In this paper, we reduce the memory consumption for end-to-end training, and manage to scale up the TAD backbone to 1 billion parameters and the input video to 1,536 frames, leading to significant detection performance. The key to our approach lies in our proposed temporal-informative adapter (TIA), which is a novel lightweight module that reduces training memory. Using TIA, we free the humongous backbone from learning to adapt to the TAD task by only updating the parameters in TIA. TIA also leads to better TAD representation by temporally aggregating context from adjacent frames throughout the backbone. We evaluate our model across four representative datasets. Owing to our efficient design, we are able to train end-to-end on VideoMAEv2-giant and achieve 75.4% mAP on THUMOS14, being the first end-to-end model to outperform the best feature-based methods.

## Prepare the pretrained VideoMAE checkpoints

Before running the experiments, please download the pretrained VideoMAE model weights (converted from original repo), and put them under the path `./pretrained/`.

|    Model     | Pretrain Dataset | Finetune Dataset |                                           Original Link                                           |                                                                  Converted Checkpoints                                                                   |
| :----------: | :--------------: | :--------------: | :-----------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  VideoMAE-S  |       K400       |       K400       | [Url](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md#:~:text=/log/-,checkpoint,-79.0) |                            [Google Drive](https://drive.google.com/file/d/1xZrJoiYCNO2pxjHo0GezIDoMo1KQPNQN/view?usp=sharing)                            |
|  VideoMAE-B  |       K400       |       K400       | [Url](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md#:~:text=/log/-,checkpoint,-81.5) | [mmaction2](https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth)  |
|  VideoMAE-L  |       K400       |       K400       | [Url](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md#:~:text=/log/-,checkpoint,-85.2) | [mmaction2](https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-229dbb03.pth) |
|  VideoMAE-H  |       K400       |       K400       | [Url](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md#:~:text=/log/-,checkpoint,-86.6) |                            [Google Drive](https://drive.google.com/file/d/1Zx-U8AZv2-P32iKZCouP4ysOgI4qRMnt/view?usp=sharing)                            |
| VideoMAEv2-g |      Hybrid      |       K710       |           [Url](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md)            |                                                                       Not Provided                                                                       |

- Note that we are not allowed to redistribute VideoMAEv2's checkpoints. You can fill out the official [request form](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md#model-weight-links), then convert the checkpoint by the following command.

```bash
python tools/model_converters/convert_videomaev2.py \
    vit_g_hybrid_pt_1200e_k710_ft.pth pretrained/vit-giant-p14_videomaev2-hybrid_pt_1200e_k710_ft_my.pth
```

## ActivityNet Results

Please refer to [README.md](../../tools/prepare_data/activitynet/README.md#download-raw-videos) to prepare the raw video of ActivityNet.

|   Backbone   | GPUs  | Setting | Frames | Img Size |  Classifier  | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |                             Config                             |                                                                                          Download                                                                                          |
| :----------: | :---: | :-----: | :----: | :------: | :----------: | :-----: | :------: | :------: | :------: | :------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  VideoMAE-S  |   4   | AdaTAD  |  768   |   160    |     CUHK     |  56.23  |  38.90   |   8.88   |  37.81   |    [config](anet/e2e_anet_videomae_s_192x4_160_adapter.py)     | [model](https://drive.google.com/file/d/1SjYsk1bPvBVvnPSy15OodnzJ45VkGWP6/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1Qv7soBzgfyEcyXNRgOxv3IHc-kVE7W2x/view?usp=sharing) |
|  VideoMAE-B  |   4   | AdaTAD  |  768   |   160    |     CUHK     |  56.72  |  39.44   |   9.54   |  38.35   |    [config](anet/e2e_anet_videomae_b_192x4_160_adapter.py)     | [model](https://drive.google.com/file/d/1_qYleQ-BC1FAjcFPBkot4Y03b04xUUtX/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1xsa34gdu-YMNAjBtqAAsaZE8gtDchRLK/view?usp=sharing) |
|  VideoMAE-L  |   4   | AdaTAD  |  768   |   160    |     CUHK     |  57.73  |  40.53   |   9.96   |  39.21   |    [config](anet/e2e_anet_videomae_l_192x4_160_adapter.py)     | [model](https://drive.google.com/file/d/1H2Wj8TTA0og1TaQtBkThYujP6daW5-Wy/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1NHC9bct2JSS2goY2pKCutsxH17g0ZoFp/view?usp=sharing) |
|  VideoMAE-H  |   4   | AdaTAD  |  768   |   160    |     CUHK     |  57.77  |  40.60   |   9.78   |  39.31   |    [config](anet/e2e_anet_videomae_h_192x4_160_adapter.py)     | [model](https://drive.google.com/file/d/1IDrkC4rIZvvk2KGht_zn_NFHZdHn9M0d/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1NYGe7TV9IMzN5SmsE9sFrrKmGGkkHUyM/view?usp=sharing) |
| VideoMAEV2-g |   4   | AdaTAD  |  768   |   160    |     CUHK     |  58.42  |  40.89   |  10.01   |  39.77   |   [config](anet/e2e_anet_videomaev2_g_192x4_160_adapter.py)    | [model](https://drive.google.com/file/d/1JLCMY14QPG98aDw0tjU83i07SvkZobDi/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1NTB59Wzxr8HpdBJPPnvOpGO0oItIbWet/view?usp=sharing) |
| VideoMAEV2-g |   8   | AdaTAD  |  768   |   224    |     CUHK     |  58.57  |  41.19   |  10.27   |  39.86   |       [config](e2e_anet_videomaev2_g_192x4_224_adapter)        | [model](https://drive.google.com/file/d/1tlv5hlAzUNHgqvq9DzAWOv3fQj6Z2ZJs/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1QTV9JzCA_wFmPvsyzzwSQ5CT3B8dIVAt/view?usp=sharing) |
| VideoMAEV2-g |   8   | AdaTAD  |  768   |   224    | InternVideo  |  61.74  |  43.17   |  10.68   |  41.85   | [config](e2e_anet_videomaev2_g_192x4_224_adapter_internvideo)  |                                                 [log](https://drive.google.com/file/d/16MpHecosa_4QkNvqjUaU07So0z374m5f/view?usp=sharing)                                                  |
| VideoMAEV2-g |   8   | AdaTAD  |  768   |   224    | InternVideo2 |  63.59  |  44.31   |  10.66   |  42.90   | [config](e2e_anet_videomaev2_g_192x4_224_adapter_internvideo2) |                                                 [log](https://drive.google.com/file/d/1uGKwVnPhIoA3zl9ZEdvT86kRnUtXt6oW/view?usp=sharing)                                                  |

- To train the model on ActivityNet, you can run the following command.

```bash
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/anet/e2e_anet_videomae_s_192x4_160_adapter.py
```

- To use the same checkpoint but test with another classifier, you can run the following command.

```bash
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/adatad/anet/e2e_anet_videomaev2_g_192x4_224_adapter_internvideo2.py --checkpoint epoch_10_cba1017a.pth
```

## THUMOS-14 Results

Please refer to [README.md](../../tools/prepare_data/thumos/README.md#download-raw-videos) to prepare the raw video of THUMOS.

|   Backbone   | GPUs  | Setting | Frames | Img Size | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |                             Config                              |                                                                                          Download                                                                                          |
| :----------: | :---: | :-----: | :----: | :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  VideoMAE-S  |   2   | AdaTAD  |  768   |   160    |  83.90  |  79.01  |  72.38  |  61.57  |  48.27  |  69.03   |   [config](thumos/e2e_thumos_videomae_s_768x1_160_adapter.py)   | [model](https://drive.google.com/file/d/1HGUBroK90KBAkFqQreAVtHCIclJh7DmM/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1sqLsgkZsPReusv1lNUOg_nqE4nJX-YnD/view?usp=sharing) |
|  VideoMAE-B  |   2   | AdaTAD  |  768   |   160    |  85.95  |  81.86  |  75.02  |  63.29  |  49.56  |  71.14   |   [config](thumos/e2e_thumos_videomae_b_768x1_160_adapter.py)   | [model](https://drive.google.com/file/d/1PFqXL4HcRv4cwqrZnhSwKjG53kEFByLs/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1uRY53OHcsxREVNHR-O-mcJyZde1XDUhe/view?usp=sharing) |
|  VideoMAE-L  |   2   | AdaTAD  |  768   |   160    |  87.17  |  83.58  |  76.88  |  66.81  |  53.13  |  73.51   |   [config](thumos/e2e_thumos_videomae_l_768x1_160_adapter.py)   | [model](https://drive.google.com/file/d/1vCbNU82TFjh0b6BRP566Jj1JHC-3qcum/view?usp=sharing)   \| [log](https://drive.google.com/file/d/147aU9TNEjxSxoVJ0S7lsYQ-mTsYHsvHK/view?usp=sharing) |
|  VideoMAE-H  |   2   | AdaTAD  |  768   |   160    |  88.42  |  84.63  |  78.72  |  69.04  |  53.95  |  74.95   |   [config](thumos/e2e_thumos_videomae_h_768x1_160_adapter.py)   | [model](https://drive.google.com/file/d/1egFK_6bLiyj1Doo0kZ3wTR5XE9JYyn1e/view?usp=sharing)   \| [log](https://drive.google.com/file/d/19cFQm4RDxGly9pWFbz1Yfxoqix-y-Stw/view?usp=sharing) |
| VideoMAEV2-g |   2   | AdaTAD  |  768   |   160    |  88.63  |  85.39  |  79.17  |  68.34  |  53.79  |  75.06   |  [config](thumos/e2e_thumos_videomaev2_g_768x1_160_adapter.py)  | [model](https://drive.google.com/file/d/1QV4a_8ulkSnf4-rQbzw58flgFGCwUT_6/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1x7giiYmP-95eXUpwpYd70TTtcttBrh-Q/view?usp=sharing) |
| VideoMAEV2-g |   2   | AdaTAD  |  1536  |   224    |  89.93  |  86.83  |  81.24  |  69.97  |  57.36  |  77.07   |  [config](thumos/e2e_thumos_videomaev2_g_768x2_224_adapter.py)  | [model](https://drive.google.com/file/d/1sANYRJE0lfbOTZJKj-1jj4S53jVLW3Wf/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1QncfW_dxJRE2nhNGL6JmRB3-SBBuv2NJ/view?usp=sharing) |
| VideoMAEV2-g |   2   | AdaTAD† |  1536  |   224    |  88.43  |  84.72  |  77.88  |  68.51  |  53.72  |  74.65   | [config](thumos/e2e_thumos_videomaev2_g_768x2_224_side_2e-4.py) | [model](https://drive.google.com/file/d/13o1d6FxEngkAMs1NpFAIpQxtchflW0Bw/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1tJEeDL7yq_pFw6YDFqLKf1eVLTokMQlq/view?usp=sharing) |

- To train the model on THUMOS, you can run the following command.

```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py
```

- To search the adapter's learning rate, or change other hyper-parameters, you can run the following command.

```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py \ 
  --cfg-options optimizer.backbone.custom.0.lr=1e-4 --id 1
```

## EPIC-KITCHENS Results


Before running the experiments, please download the EPIC-pretrained VideoMAE's weights, and put them under the path `./pretrained/`.

|   Model    | Pretrain Dataset | Finetune Dataset |                                            Checkpoints                                             |
| :--------: | :--------------: | :--------------: | :------------------------------------------------------------------------------------------------: |
| VideoMAE-L |   InternVideo1   |    EPIC-Noun     | [Google Drive](https://drive.google.com/file/d/1_GpDIVFb4Aj-EPBEU9ekK_pMc0NVWvN2/view?usp=sharing) |
| VideoMAE-L |   InternVideo1   |    EPIC-Verb     | [Google Drive](https://drive.google.com/file/d/1W-gg0aX1OQVbPU3U7AZJcP2bSikPUEVA/view?usp=sharing) |

Please refer to [README.md](../../tools/prepare_data/epic/README.md#download-raw-videos) to prepare the raw video of EPIC-Kitchens.

| Subset |   Backbone    | GPUs  | Setting | Frames | Img Size | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | ave. mAP |                             Config                              |                                                                                          Download                                                                                          |
| :----: | :-----------: | :---: | :-----: | :----: | :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  Noun  | VideoMAE-Noun |   2   | AdaTAD  | 768x8  |   160    |  33.88  |  32.41  |  30.58  |  27.66  |  22.67  |  29.44   | [config](epic/e2e_epic_videomae_l_ft_768x8_160_adapter_noun.py) | [model](https://drive.google.com/file/d/1IENRReN02NOnm_RJBVO1bPU5Nce3Hyyh/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1bQbrCh5_0O6hLljCc8QQ4N8UW6tRIKIz/view?usp=sharing) |
|  Verb  | VideoMAE-Verb |   2   | AdaTAD  | 768x8  |   160    |  33.02  |  32.43  |  30.51  |  27.80  |  24.69  |  29.69   | [config](epic/e2e_epic_videomae_l_ft_768x8_160_adapter_verb.py) | [model](https://drive.google.com/file/d/123ENCu0A677rT1izN-GaWlNAqte1i-gt/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1NLpMXon9EGgOnaVOgba3Wxogn9TT8-ar/view?usp=sharing) |

- To train the model on EPIC-Kitchens, you can run the following command.

```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_noun.py
```

## Ego4D-MQ Results


Before running the experiments, please download the InternVideo1-MQ weights, and put them under the path `./pretrained/`.

|      Model      | Pretrain Dataset  |   Finetune Dataset    |                                                Original Link                                                |                                       Converted Checkpoints                                        |
| :-------------: | :---------------: | :-------------------: | :---------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------: |
| InternVideo1-MQ | InternVideo1-K700 | Ego4D-Verb + Ego4D-MQ | [Url](https://github.com/OpenGVLab/ego4d-eccv2022-solutions/tree/main#:~:text=%2D-,Download,-UniFormer%2DB) | [Google Drive](https://drive.google.com/file/d/1r4e6GDLg0F8VVGPwYUiijO6ECH63t02z/view?usp=sharing) |

Please refer to [README.md](../../tools/prepare_data/ego4d/README.md#download-raw-videos) to prepare the raw video of Ego4D-MQ.

|    Backbone     | GPUs  | Setting | Frames | Img Size | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | ave. mAP |                              Config                               |                                                                                          Download                                                                                          |
| :-------------: | :---: | :-----: | :----: | :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :---------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| InternVideo1-MQ |   2   | AdaTAD  | 1800x4 |   192    |  33.69  |  31.19  |  28.37  |  26.12  |  22.67  |  28.41   | [config](epic/e2e_ego4d_internvideo_1800x4_192_adapter_lr4e-4.py) | [model](https://drive.google.com/file/d/16MkRDEFLVo4xs0N0yEpw7TPB_h5ZlGXL/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1u62mkktrsZee88fzJ7h4gbSuoQuJ-5aa/view?usp=sharing) |

- To train the model on Ego4D-MQ, you can run the following command.

```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/ego4d/e2e_ego4d_internvideo_1800x4_192_adapter_lr4e-4.py
```

## Multi-THUMOS Results

Please refer to [README.md](../../tools/prepare_data/multi-thumos/README.md#download-raw-videos) to prepare the raw video of THUMOS.

|   Backbone   | GPUs  | Setting | Frames | Img Size | mAP@0.2 | mAP@0.5 | mAP@0.7 | ave. mAP (0.1:0.9:0.1) |                                  Config                                  |                                                                                          Download                                                                                          |
| :----------: | :---: | :-----: | :----: | :------: | :-----: | :-----: | :-----: | :--------------------: | :----------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  VideoMAE-S  |   2   | AdaTAD  |  768   |   160    |  61.34  |  46.74  |  26.88  |         40.77          |  [config](multi_thumos/e2e_multithumos_videomae_s_768x1_160_adapter.py)  | [model](https://drive.google.com/file/d/1Fto6mSOPDPiJ_aZBGj4hJAlB80jIQHoX/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1wkQMsmMPojTVcbAJc6F1JLeHj0V0qV50/view?usp=sharing) |
|  VideoMAE-B  |   2   | AdaTAD  |  768   |   160    |  63.90  |  48.74  |  28.72  |         42.76          |  [config](multi_thumos/e2e_multithumos_videomae_b_768x1_160_adapter.py)  | [model](https://drive.google.com/file/d/1BFqLFYThYP3tXMglMKvIQXov9jOWfDUz/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1o2FubV2KOTKIHpmqNRzKu0FOp7KEUZAf/view?usp=sharing) |
|  VideoMAE-L  |   2   | AdaTAD  |  768   |   160    |  66.06  |  51.80  |  31.73  |         45.15          |  [config](multi_thumos/e2e_multithumos_videomae_l_768x1_160_adapter.py)  | [model](https://drive.google.com/file/d/1rgp77wKWVWgJlC2yM8qhHCQxSpFb4tgR/view?usp=sharing)   \| [log](https://drive.google.com/file/d/19EgbRxgm66uuDyM-w6mpQtYuHdO_LeRS/view?usp=sharing) |
|  VideoMAE-H  |   2   | AdaTAD  |  768   |   160    |  67.20  |  52.99  |  32.70  |         46.02          |  [config](multi_thumos/e2e_multithumos_videomae_h_768x1_160_adapter.py)  | [model](https://drive.google.com/file/d/1Si-_eP0aEhJZ7uVcHvJtGnetmdkeH_Qs/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1xtixx9mtA0GBDufWml5t9cvOTN2T9_Od/view?usp=sharing) |
| VideoMAEV2-g |   2   | AdaTAD  |  768   |   160    |  68.23  |  53.87  |  33.03  |         46.74          | [config](multi_thumos/e2e_multithumos_videomaev2_g_768x1_160_adapter.py) | [model](https://drive.google.com/file/d/15ri0cDo2dRHF1C3XnpbgpaiMF0TyhFD3/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1_L4sIjdz0BVfI5xNuLyRhUB5j715WkjV/view?usp=sharing) |
| VideoMAEV2-g |   2   | AdaTAD  |  1536  |   224    |  71.11  |  55.83  |  34.86  |         48.73          | [config](multi_thumos/e2e_multithumos_videomaev2_g_768x2_224_adapter.py) | [model](https://drive.google.com/file/d/1GALzlQN1w-YjMbphpRmMd4qqqKCKF2cq/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1wvKTaj03gjLhPO7oFk8ePT9OEdN-k3b2/view?usp=sharing) |

- To train the model on Multi-THUMOS, you can run the following command.

```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/multi_thumos/e2e_multithumos_videomae_s_768x1_160_adapter.py
```



## Citation

```latex
@inproceedings{liu2023end,
  title={End-to-End Temporal Action Detection with 1B Parameters Across 1000 Frames},
  author={Liu, Shuming and Zhang, Chen-Lin and Zhao, Chen and Ghanem, Bernard},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```