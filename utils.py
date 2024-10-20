import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import DepthDataset

from hyperparameters import *

#A.Rotate(limit = 35, p = 1.0),
#A.HorizontalFlip(p = 0.5),
#A.VerticalFlip(p = 0.5),

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(train_transform, test_transform, train_transform_gt, test_transform_gt):

    train_ds = DepthDataset(img_dir=TRAIN_IMG_DIR, gt_dir=TRAIN_GT_DIR, transform=train_transform, transform_gt=train_transform_gt)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)

    test_ds = DepthDataset(img_dir=TEST_IMG_DIR, gt_dir=TEST_GT_DIR, transform=test_transform, transform_gt=test_transform_gt)

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)

    return train_loader, test_loader

def check_accuracy(loader, model, device):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)
            y = y.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(x))
            preds_binary = (preds > 0.5).float()
            
            num_correct += (preds_binary == y).sum()
            num_pixels += torch.numel(preds_binary)

            dice_score += (2 * (preds_binary * y).sum()) / (
                (preds_binary + y).sum() + 1e-8
            )

    accuracy = num_correct / num_pixels * 100
    dice_score = dice_score / len(loader)
    print(f"Accuracy: {accuracy:.2f}%, Dice Score: {dice_score:.4f}")

    model.train()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    
    with torch.no_grad():
        for idx, (x, _) in enumerate(loader):
            x = x.float().to(device)
            preds = torch.sigmoid(model(x))
            
            transform = transforms.ToPILImage()
            for i in range(preds.shape[0]):
                img = preds[i].squeeze(0).cpu()
                img = transform(img)
                
                img.save(f"{folder}/pred_{idx}_{i}.png")
    
    model.train()