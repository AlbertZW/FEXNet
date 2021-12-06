# FEXNet: Foreground Extraction Network for Human Action Recognition
## Overview
We release the PyTorch code of [FEXNet](https://ieeexplore.ieee.org/abstract/document/9509412). 

![](./images/Global_structure.png)
![](./images/SS.png)
![](./images/FE.png)

The code is majorly based on [TSM](https://github.com/mit-han-lab/temporal-shift-module). The proposed **Foreground EXtraction** blocks contains **Scene Segregation** and **Foreground Enhancement** modules to extract foreground features in different aspects as shown in the figures above.

The detailed data pre-processing and preperation strategies follow the settings of [TSM](https://github.com/mit-han-lab/temporal-shift-module).

## Prerequisites
The code is built with following libraries:

- [PyTorch](https://pytorch.org/) 1.5.0 or higher
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [tqdm](https://github.com/tqdm/tqdm.git)
- [scikit-learn](https://scikit-learn.org/stable/)

## Testing


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
