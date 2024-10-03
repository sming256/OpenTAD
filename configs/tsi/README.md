# TSI

> [TSI: Temporal Scale Invariant Network for Action Proposal Generation](https://openaccess.thecvf.com/content/ACCV2020/papers/Liu_TSI_Temporal_Scale_Invariant_Network_for_Action_Proposal_Generation_ACCV_2020_paper.pdf)  
> Shuming Liu, Xu Zhao, Haisheng Su, and Zhilan Hu

<!-- [ALGORITHM] -->

## Abstract

Despite the great progress in temporal action proposal generation, most state-of-the-art methods ignore the impact of action scales and the performance of short actions is still far from satisfaction. In this paper, we first analyze the sample imbalance issue in action proposal generation, and correspondingly devise a novel scale-invariant loss function to alleviate the insufficient learning of short actions. To further achieve proposal generation task, we adopt the pipeline of boundary evaluation and proposal completeness regression, and propose the Temporal Scale Invariant network. To better leverage the temporal context, boundary evaluation module generates action boundaries with high-precision-assured global branch and high-recall-assured local branch. Simultaneously, the proposal evaluation module is supervised with introduced scale-invariant loss, predicting accurate proposal completeness for different scales of actions. Comprehensive experiments are conducted on ActivityNet-1.3 and THUMOS14 benchmarks, where TSI achieves state-of-the-art performance. Especially, AUC performance of short actions is boosted from 36.53% to 39.63% compared with baseline.

## Results and Models

**ActivityNet-1.3** with CUHK classifier.

| Features | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |        Config         |                                                                                          Download                                                                                          |
| :------: | :-----: | :------: | :------: | :------: | :-------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   TSN    |  49.76  |  33.01   |   9.03   |  32.88   | [config](anet_tsn.py) | [model](https://drive.google.com/file/d/1urZz7O0kUyKFz77nKLfnwUPKpWIjJMI5/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1nBsbeckbhJhy3XRBUfcOPzc50Tlb3MXp/view?usp=sharing) |
|   TSP    |  52.44  |  35.57   |   9.80   |  35.36   | [config](anet_tsp.py) | [model](https://drive.google.com/file/d/1HXUnrl2x4IFPFVKIFiRtEm_qpArMV2Fl/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1IqrksCpXYWecUMPwpyL4u42wrlcpn-6N/view?usp=sharing) |

Use above checkpoints to evaluate the recall performance:

| Features | AR@1  | AR@5  | AR@10 | AR@100 |  AUC  |            Config            |                                                                                          Download                                                                                          |
| :------: | :---: | :---: | :---: | :----: | :---: | :--------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   TSN    | 32.53 | 49.64 | 57.18 | 76.45  | 68.31 | [config](anet_tsn_recall.py) | [model](https://drive.google.com/file/d/1urZz7O0kUyKFz77nKLfnwUPKpWIjJMI5/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1aJpSzeRnWFcvuyufzwNhzCKg2tKHZkES/view?usp=sharing) |
|   TSP    | 34.12 | 52.03 | 59.45 | 77.20  | 69.72 | [config](anet_tsp_recall.py) | [model](https://drive.google.com/file/d/1HXUnrl2x4IFPFVKIFiRtEm_qpArMV2Fl/view?usp=sharing)   \| [log](https://drive.google.com/file/d/14uWV75yTnXV3UEOeD8B4q2_9qbct4RVE/view?usp=sharing) |


**THUMOS-14** with UtrimmedNet classifier

| Features | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |         Config          |                                                                                          Download                                                                                          |
| :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :---------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   TSN    |  60.37  |  54.38  |  46.14  |  37.05  |  25.81  |  44.75   | [config](thumos_tsn.py) | [model](https://drive.google.com/file/d/1wC1Plyb526FARC4_0PJXZYbCxt7bqDCF/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1dp9cv2aHk5Z00puZkKBN1ozVKEnNwD27/view?usp=sharing) |
|   I3D    |  62.56  |  57.00  |  50.22  |  40.18  |  30.17  |  48.03   | [config](thumos_i3d.py) | [model](https://drive.google.com/file/d/1TBPf5ZrYD0LzfZwX627j-iGmZ3SI_tn7/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1n8DFPno0UVPDFfwp6qt5a0jqAdKCGrYu/view?usp=sharing) |


## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TSI on ActivityNet dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/tsi/anet_tsp.py
```

For more details, you can refer to the Training part in the [Usage](../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TSI on ActivityNet dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/tsi/anet_tsp.py --checkpoint exps/anet/tsi_tsp_128/gpu1_id0/checkpoint/epoch_9.pth
```

To test the recal performance:

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/tsi/anet_tsp_recall.py --checkpoint exps/anet/tsi_tsp_128/gpu1_id0/checkpoint/epoch_9.pth
```

For more details, you can refer to the Test part in the [Usage](../../docs/en/usage.md).

## Citation

```latex
@inproceedings{liu2020tsi,
  title={TSI: Temporal Scale Invariant Network for Action Proposal Generation},
  author={Liu, Shuming and Zhao, Xu and Su, Haisheng and Hu, Zhilan},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  year={2020}
}
```