# README

## Setting Up the Environment
To run the code in this repository, you need to create a Conda environment with the following dependencies:

### Create Conda Environment
```sh
conda create -n tos_env python=3.9 -y
conda activate tos_env
```

### Install Required Packages
```sh
pip install torch==2.0.0 torchvision==0.15.1 mmsegmentation==0.27.0 mmengine==0.10.1
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html
```

### CUDA Compatibility
Ensure you have **CUDA 11.7** installed and configured correctly. The required CUDA dependencies should be included with the above packages, but verify with:
```sh
nvcc --version
```

## Running the Code
To train a model, use the following command structure:
```sh
python tools/train.py --config <path_to_config> --work-dir <path_to_output_directory>
```

### Example Usage
For training with the `deeplabv3plus_resnet_tos` configuration:
```sh
python tools/train.py --config ./configs/tos/deeplabv3plus_resnet_tos.py --work-dir ./work_dirs/deeplabv3plus_resnet
```

### Configurations Available
This repository includes multiple model configurations:
- `configs/tos/deeplabv3_unet_tos.py`
- `configs/tos/deeplabv3plus_resnet_tos.py`
- `configs/tos/swin_l_tos.py`
- `configs/tos/unet_tos.py`

The user should select one of the available configuration files and specify the working directory accordingly.

## Notes
- Ensure that you have a compatible GPU with CUDA 11.7 support.
- If additional dependencies are required, install them using `pip install <package_name>`.
- For further details, refer to the official documentation of the libraries used:
  - [PyTorch](https://pytorch.org/)
  - [Torchvision](https://github.com/pytorch/vision)
  - [MMSegmentation](http://github.com/open-mmlab/mmsegmentation)
  - [MMCV](https://github.com/open-mmlab/mmcv)
  - [MMEngine](https://github.com/open-mmlab/mmengine)

## Citation
If you find our project helpful, please consider citing our work. 

```
@inproceedings{sahin2023tinypedseg,
  title={TinyPedSeg: A Tiny Pedestrian Segmentation Benchmark for Top-Down Drone Images},
  author={Sahin, Yusuf H and Abdinli, Elvin and Ayd{\i}n, M Arda and Unal, Gozde},
  booktitle={2023 18th International Conference on Machine Vision and Applications (MVA)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

## Acknowledgments
This implementation builds upon [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). We gratefully acknowledge their valuable contributions.
