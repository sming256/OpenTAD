# TadTR

> [End-to-end Temporal Action Detection with Transformer](https://arxiv.org/abs/2106.10271)  
> Xiaolong Liu, Qimeng Wang, Yao Hu, Xu Tang, Shiwei Zhang, Song Bai, Xiang Bai
<!-- [ALGORITHM] -->

> [An Empirical Study of End-to-End Temporal Action Detection](https://arxiv.org/abs/2204.02932)  
> Xiaolong Liu, Song Bai, Xiang Bai
<!-- [ALGORITHM] -->

## Results and Models

**THUMOS-14**

| Feature | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |         Config          |                                                                                          Download                                                                                          |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :---------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   I3D   |  71.90  |  67.29  |  59.00  |  48.34  |  34.61  |  56.23   | [config](thumos_i3d.py) | [model](https://drive.google.com/file/d/1m0WXxLwF4zcAfd7Gdr3GtwV0j6K7aTYc/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1qAt2h_fnhuzfky4Jmw-vxUqUbbNwRY1c/view?usp=sharing) |

|    Backbone    | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |                      Config                      |                                                                                          Download                                                                                          |
| :------------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :----------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| E2E-SlowFasR50 |  64.47  |  59.49  |  53.11  |  44.08  |  33.50  |  50.93   | [config](e2e_thumos_tadtr_slowfast50_sw128s6.py) | [model](https://drive.google.com/file/d/16iYYzhcstf91MIS-86UIvagjcpCd2bsr/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1GuoxGfeHCScHBOUiy4KhMmNPCO2vRMaS/view?usp=sharing) |


## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TadTR on THUMOS dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/tadtr/thumos_i3d.py
```

For more details, you can refer to the Training part in the [Usage](../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TadTR on THUMOS dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/tadtr/thumos_i3d.py --checkpoint exps/thumos/tridet_i3d/gpu1_id0/checkpoint/epoch_14.pth
```

For more details, you can refer to the Test part in the [Usage](../../docs/en/usage.md).


## Citation

```latex
@article{liu2022end,
  title={End-to-end temporal action detection with transformer},
  author={Liu, Xiaolong and Wang, Qimeng and Hu, Yao and Tang, Xu and Zhang, Shiwei and Bai, Song and Bai, Xiang},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={5427--5441},
  year={2022},
  publisher={IEEE}
}

@inproceedings{liu2022empirical,
  title={An empirical study of end-to-end temporal action detection},
  author={Liu, Xiaolong and Bai, Song and Bai, Xiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20010--20019},
  year={2022}
}
```