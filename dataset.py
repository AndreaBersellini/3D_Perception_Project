import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from hyperparameters import *

class DepthDataset(Dataset):
    def __init__(self, img_dir, gt_dir, transform = None, transform_gt = None):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.transform_gt = transform_gt
        self.images = os.listdir(img_dir)
        self.gts = os.listdir(gt_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        gt_path = os.path.join(self.gt_dir, self.gts[index])

        image = Image.open(img_path)
        ground_truth = Image.open(gt_path)

        match (IN_CHANNELS, OUT_CHANNELS):
            case (1, 1):
                image = image.convert("L")
                if self.transform is not None and self.transform_gt is not None:
                    image = self.transform_gt(image)
                    ground_truth = self.transform_gt(ground_truth)
            case (3, 1):
                if self.transform is not None and self.transform_gt is not None:
                    image = self.transform(image)
                    ground_truth = self.transform_gt(ground_truth)

        return image, ground_truth

if __name__ == "__main__":
    dt = DepthDataset(TRAIN_IMG_DIR, TRAIN_GT_DIR)

    data = dt.__getitem__(0)

    img = data[0]
    depth = data[1]

    plt.imshow(img, cmap = 'gray')
    plt.title('Image')
    plt.axis('off')
    plt.show()

    plt.imshow(depth, cmap = 'gray')
    plt.title('Disparity Map')
    plt.axis('off')
    plt.show()