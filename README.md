# Neural Encoder

This repository contains the code for the Neural Encoder project. The goal of this project is to provide a robust and efficient neural encoding framework. Users can add their own components into this framework conveniently (i.e. add new network models into `./model/net/new_model.py`) and regist this new model with ```@MODELS.register_module()```. The registering mechanism of this project is referred to [mmcv](https://github.com/open-mmlab/mmcv), users can find more detailed descriptions in their project.

## Getting Started

### Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (Anaconda or Miniconda)
- This project is tested on Windows 11.

### Setting Up the Environment

1. Download or clone the repository:
    Users can download the repository directly from Zenodo.
    Or can clone the repository from Github:
    ```bash
    git init
    git clone https://github.com/zhurz21/ultra-generalized_Neural_Encoder
    cd ultra-generalized_Neural_Encoder
    ```

2. You can set up a conda environment with all dependencies using `environment.yml`:
    First, change the `prefix` in `environment.yml` to your conda environment path (`...\anaconda3\envs\neural-encoder`). 
    
    Then, using the following commands.

    ```bash
    conda env create -f environment.yml
    conda activate neural-encoder
    ```

    The pretrained model weights for neural encoder can be downloaded from [here](). To run with the pretrained model, please place the downloaded contents in the `WH/` folder.

## Project Structure

- `config/`: Contains configuration files for training and testing.
- `data_operator/`: Contains scripts for data generating and preprocessing.
- `ini_phase/`: Contains the pregenerated initial phases.
- `model/`: Contains the network models and physical model for wave optics.
- `Test dataset/`: Contains the 2D test dataset for evaluation.
- `Test dataset 3D/`: Contains the RGBD dataset for evaluation.
- `Test dataset other3D/`: Contains the 3D test data for Light Fields, polygon meshes,and Gaussian splattings.
- `tools/`: Contains the renderer class for the model evaluation and result saveing.
- `utils/`: General utility functions and helpers.
- `WH/`: Directory for storing pretrained model weights and inference results.

## Running the test

Before testing the neural encoder, please make sure that the following materials are in the correct locations:
- The pretrained neural encoder models are placed in the `WH/` folder.
- The initial phases (if you want to test your customized initial phases) are placed in the `ini_phase/` folder.
- The 2D test dataa are placed in the `Test dataset/` folder.
- The 3D test data are place in the `Test dataset 3D/` folder.

### Testing the Model

To test a trained neural encoder, run the following command:
- 2D hologram generation
```bash
python train_encoder.py --exp_name UNet_filter_x4_y4_AdamW_bs1_DDP_rectfilter_2w --test_only
```
- 3D layer-based hologram generation
```bash
python train_encoder.py --exp_name UNet_filter_x4_y4_AdamW_bs1_DDP_rectfilter_2w --test_only --test_dimension 3D --data_type layer --test_path "Test dataset 3D"
```
- 3D point-source-based hologram generation
```bash
python train_encoder.py --exp_name UNet_filter_x4_y4_AdamW_bs1_DDP_rectfilter_2w --test_only --test_dimension 3D --data_type point --test_path "Test dataset 3D"
```
- 3D polygon-based hologram generation
```bash
python eval_other3D.py --config config/UNet/UNet_mesh_config.py --exp_name UNet_filter_x4_y4_AdamW_bs1_DDP_rectfilter_2w --data_type mesh --test_path "Test dataset other3D"
```
- 3D Light Field-based hologram generation
```bash
python eval_other3D.py --config config/UNet/UNet_LF_parallex_config.py --exp_name UNet_filter_x4_y4_AdamW_bs1_DDP_rectfilter_2w --data_type LF --test_path "Test dataset other3D"
```
- 3D Guassian-based hologram generation
```bash
python eval_other3D.py --config config/UNet/UNet_Gsplat_config.py --exp_name UNet_filter_x4_y4_AdamW_bs1_DDP_rectfilter_2w --data_type GS --test_path "Test dataset other3D"
```
- 3D end-to-end hologram generation
```bash
python train_encoder.py --exp_name UNet_filter_x4_y4_AdamW_bs1_DDP_rectfilter_2w --test_only --test_dimension 3D --data_type end-to-end --test_path "Test dataset 3D"
```
You can modify the configuration file `config/*/*_config.py` to change the testing parameters, such as diffraction distance (`meta_info['prop_dist']`), wavelength (`meta_info['channel']`), resolution (`data['slm_res']`), initial phase type (`data['ini_phase_type']`), among others.

### Train a new Model
To train a new model, run the following command:
```bash
python train_encoder.py --exp_name test
```
You can modify the configuration file `config/*/*_config.py` to change the training parameters (such as wavelength, resolution, pixel size, and initial phase map).

## Config File Structure

The configuration files `config/*/*_config.py` are structured into several sections, each containing specific parameters for different aspects of the neural encoder. Below is a brief description of each section:

### Meta Information
- `meta_info`: Contains general settings such as random seed, epoch numbers, batch sizes, feature size, wavelength, propagation distance, and learning rate scheduler.

### Dataset Information
- `data`: Defines parameters related to the dataset, including data dimensions, types, initial phase settings, Fourier basis parameters, SLM resolution, ROI resolution, and test paths.

### Model Information
- `model`: Specifies the neural network model configuration, including model name, input/output channels, network depth, normalization method, up-sampling method, dropout settings, and forward/backward diffraction methods.

### Calibration Hologram
- `cali`: Contains settings for generating calibration holograms, such as the number of circles and circle radius.

### Optimizer Information
- `optimizer`: Defines the optimizer settings, including the optimizer name, learning rate, and learning rate scheduler parameters.

Each section is designed to be easily modifiable to suit different experimental setups and requirements.

## Contributing

We welcome contributions to improve the Neural Encoder project. Please fork the repository and create a pull request with your changes.


## Contact

If you have any questions, please contact

Runze Zhu, zhurz21@mails.tsinghua.edu.cn

Hao Zhang, haozhang274@tsinghua.edu.cn
