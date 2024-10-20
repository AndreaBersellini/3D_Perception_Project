import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEARNING_RATE = 1e-4 # 1e-4 Original
BATCH_SIZE = 32
NUM_EPOCHS = 30
NUM_WORKERS = 2

PIN_MEMORY = True
LOAD_MODEL = False

IMAGE_SIZE = (128, 128)
IN_CHANNELS = 1
OUT_CHANNELS = 1

TRAIN_IMG_DIR = '../Dataset/Train/IMAGES'
TRAIN_GT_DIR = '../Dataset/Train/GT'
TEST_IMG_DIR = '../Dataset/Test/IMAGES'
TEST_GT_DIR = '../Dataset/Test/GT'