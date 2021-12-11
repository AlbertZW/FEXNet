# FEXNet: Foreground Extraction Network for Human Action Recognition
## Overview
We release the PyTorch code of [FEXNet](https://ieeexplore.ieee.org/abstract/document/9509412). 

![](./images/Global_structure.png)

The code is majorly based on [TSM](https://github.com/mit-han-lab/temporal-shift-module). The global structure of FEXNet is shown in the figure above. The proposed **Foreground EXtraction** blocks contains **Scene Segregation** and **Foreground Enhancement** modules to extract foreground features in different aspects as below.

Detailed structure of **SS Module**:
![](./images/SS.png)

Detailed structure of **FE Module**:
![](./images/FE.png)

The detailed data pre-processing and preperation strategies follow the settings of [TSM](https://github.com/mit-han-lab/temporal-shift-module).

## Prerequisites
The code is built with following libraries:

- [PyTorch](https://pytorch.org/) 1.5.0 or higher
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [tqdm](https://github.com/tqdm/tqdm.git)
- [scikit-learn](https://scikit-learn.org/stable/)

## Testing
```testing.sh``` is provided to test the pretrained models. The pretrained model for dataset ```something-something-V1``` can be found in the folder ```./checkpoint```.

```
python test_models.py something \
      --weights=./checkpoint/TSM_something_RGB_resnet50_shift6_blockres_FEX_avg_segment8_e50/ckptbest.pth \
      --test_segments=8 --test_crops=3 --batch_size 32 --twice_sample --full_res
```

The parameter setting also follows the strategy of [TSM](https://github.com/mit-han-lab/temporal-shift-module). Notably, ```--full_res``` implicates that the sampled frames for testing keep the full resolution and usually obtain higher accuracies.

## Training


## Citation
```
@article{shen2021fexnet,
  title={FEXNet: Foreground Extraction Network for Human Action Recognition},
  author={Shen, Zhongwei and Wu, Xiao-Jun and Xu, Tianyang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2021},
  publisher={IEEE}
}
```
