import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from carvana_dataset import CarvanaDataset
from unet import UNet

def pred_show_image_grid(data_path, model_path, device):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    image_dataset = CarvanaDataset(data_path, test=True)
    images = [] # Image Data
    orig_masks = [] # Ground Truth Masks
    pred_masks = [] # Predicted Masks
    
    for img, orig_mask in image_dataset:
        img = img.float().to(device)
        img = img.unsqueeze(0)
        
        pred_mask = model(img)
        
        img = img.squeeze(0).cpu().detach()
        img = img.permute(1, 2, 0)
        
        pred_mask = pred_mask.squeeze(0).cpu().detach()
        pred_mask = pred_mask.permute(1, 2, 0)
        pred_mask[pred_mask < 0] = 0    # Setting pixels to 0 when pred_mask < 0
        pred_mask[pred_mask > 0] = 1    # Setting pixels to 1 when pred_mask > 0
        
        orig_mask = orig_mask.cpu().detach()
        orig_mask = orig_mask.permute(1, 2, 0)
        
        images.append(img)
        orig_masks.append(orig_mask)
        pred_masks.append(pred_mask)
        
    images.extend(orig_masks)
    images.extend(pred_masks)
    fig = plt.figure()
    for i in range(1, 3*len(image_dataset)+1):
        fig.add_subplot(3, len(image_dataset), i)
        plt.imshow(images[i-1], cmap='gray')
    plt.show()
    
def single_image_inference(image_path, model_path, device):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device), weights_only=False))
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    img = transform(Image.open(image_path)).float().to(device)
    img = img.unsqueeze(0)  # Adding batch dimension so it can be given as input to model.
    
    pred_mask = model(img)
    
    img = img.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0)
    
    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1, 2, 0)
    pred_mask[pred_mask < 0] = 0
    pred_mask[pred_mask > 0] = 1
    
    fig = plt.figure()
    for i in range(1, 3):
        fig.add_subplot(1, 2, i)
        if i==1:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(pred_mask, cmap='gray')
    plt.show()
    
    
if __name__ == '__main__':
    SINGLE_IMAGE_PATH = './data/manual_test/03a857ce842d_4.jpg'
    DATA_PATH = './data'
    MODEL_PATH = './models/unet.pt'
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    pred_show_image_grid(DATA_PATH, MODEL_PATH, device)
    single_image_inference(SINGLE_IMAGE_PATH, MODEL_PATH, device)
    
        