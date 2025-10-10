import os

# path to saving models
models_dir = 'models'

# path to saving loss plots
losses_dir = 'losses'

# path to the data directories
#data_dir = '/home/vision/users/cgh/denoising/unet12/unet12_zhuanyi/FLIR'
data_dir = '/home/vision/users/cgh/denoising/unet12/unet+att+mask+wavelet/BPD'
#data_dir = '/home/vision/users/cgh/denoising/unet12/unet12_xiaorong/DV'
#data_dir = '/home/vision/users/cgh/denoising/unet13/IRE3'
train_dir = 'train'
# test_dir = 'test'
val_dir = 'val'
imgs_dir = 'imgs'
noisy_dir = 'noisy'
debug_dir = 'debug'


# depth of UNet 
# depth = 4 # try decreasing the depth value if there is a memory error

# text file to get text from
txt_file_dir = 'shitty_text.txt'

# maximun number of synthetic words to generate
num_synthetic_imgs = 18000
train_percentage = 0.8

resume = True  # False for trainig from scratch, True for loading a previously saved weight
#data_weight = 'trained_weights.pth'
ckpt='/home/vision/users/cgh/denoising/unet13/models/model_epoch1073.pth' # model file path to load the weights from, only useful when resume is True
lr = 1e-5          # learning rate
epochs = 500     # epochs to train for
test_img_size = (512,640)

# batch size for train and val loaders
batch_size = 2 # try decreasing the batch_size if there is a memory error

# log interval for training and validation
log_interval = 25
# test_dir = os.path.join(data_dir, test_dir, noisy_dir)
test_dir = '/home/vision/users/cgh/denoising/unet13/image2'
res_dir = 'results'
test_bs = 2

