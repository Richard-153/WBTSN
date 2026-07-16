# WBTSN

This repository provides the research code for WBTSN, a wavelet-based two-stage network for blind-pixel compensation in infrared images. The model is designed to compensate defective infrared sensor pixels while preserving local structural details through Haar wavelet decomposition, high-frequency feature guidance, and residual refinement.

This release is intended to support academic reference, code inspection, and independent re-implementation of the method described in our manuscript.

## Repository Structure

```text
.
|-- config_stage1.py       # Paths and hyperparameters for Stage I training/testing
|-- config_stage2.py       # Paths and hyperparameters for Stage II residual correction
|-- datasets_stage1.py     # Dataset loader for Stage I
|-- datasets_stage2.py     # Dataset loader for Stage II
|-- model.py               # WBTSN wavelet U-Net and wavelet modules
|-- train_stage1.py        # Stage I training script
|-- train_stage2.py        # Stage II residual-correction training script
|-- test_stage1.py         # Stage I inference script
|-- test_stage2.py         # Full two-stage inference script
|-- requirements.txt       # Main Python dependencies
`-- README.md
```

## Environment

The code was developed with Python and PyTorch. A CUDA-enabled GPU is recommended for training.

Main dependencies:

```text
python
torch
torchvision
opencv-python
numpy
matplotlib
tqdm
scikit-image
pytorch-wavelets
```

Example installation:

```bash
pip install -r requirements.txt
```

Please install a PyTorch version compatible with your CUDA driver. See the official PyTorch installation guide if GPU acceleration is required.

## Data Format

The training and validation loaders expect paired grayscale images arranged as follows:

```text
DATA_ROOT/
|-- train/
|   |-- imgs/      # Ground-truth clean infrared images
|   `-- noisy/     # Corresponding defective/blind-pixel images
|-- val/
|   |-- imgs/
|   `-- noisy/
`-- test/
    |-- imgs/      # Optional ground truth for metric computation
    `-- noisy/     # Images for inference
```

Supported image formats include `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and `.tiff`. The code assumes grayscale input images. During loading, images are read as single-channel images and converted to PyTorch tensors.

## Configuration

Before training or testing, edit `config_stage1.py` and `config_stage2.py`.

For Stage I, set:

```python
data_dir = "/path/to/DATA_ROOT"
resume = False
ckpt = ""
lr = 1e-5
epochs = 200
batch_size = 2
```

For Stage II, set the Stage I checkpoint path:

```python
data_dir = "/path/to/DATA_ROOT"
generator_ckpt_path = "models/model_epoch200.pth"
epochs = 200
batch_size = 4
```

If you want to run inference from checkpoints, set:

```python
# config_stage1.py
ckpt = "models/model_epoch200.pth"

# config_stage2.py
generator_ckpt_path = "models/model_epoch200.pth"
test_correction_model_path = "models_stage2/correction_net_epoch_200.pth"
```

Pretrained weights and checkpoints are not provided in this repository. Please use checkpoints trained on your own data or another checkpoint that you are authorized to use.

## Training

Train Stage I:

```bash
python train_stage1.py
```

The script saves Stage I checkpoints to:

```text
models/
```

Then set `generator_ckpt_path` in `config_stage2.py` to the trained Stage I checkpoint and train Stage II:

```bash
python train_stage2.py
```

The script saves Stage II checkpoints to:

```text
models_stage2/
```

Stage I uses an image-domain reconstruction loss together with Haar wavelet sub-band losses. Stage II uses a frozen Stage I model and trains a residual correction network with image-domain residual constraints and adversarial refinement.

## Inference

To run Stage I inference, set `ckpt`, `test_dir`, and `res_dir` in `config_stage1.py`, then run:

```bash
python test_stage1.py
```

To run full two-stage inference, set `generator_ckpt_path`, `test_correction_model_path`, `test_dir`, and `test_res_dir` in `config_stage2.py`, then run:

```bash
python test_stage2.py
```

If `test_gt_dir` contains ground-truth images with matching filenames, `test_stage2.py` also reports average PSNR and SSIM.

## Availability Notice

Due to laboratory data confidentiality and security regulations, the infrared datasets used in the paper and the trained model weights cannot be publicly released. This repository is therefore a code-only release. The training data, validation data, test data, private infrared images, pretrained weights, and training checkpoints will not be publicly distributed.

## Maintenance Policy

This repository is released as an archival research code release accompanying the manuscript. The code is provided as-is for academic use. We may not be able to provide ongoing maintenance, feature updates, environment debugging, dataset access support, or pretrained-weight access support.

Issues and pull requests may be reviewed when time permits, but no regular maintenance schedule is planned.

## Contact

For essential questions about the code release, please contact:

```text
yebidaxiong2025@163.com
```

