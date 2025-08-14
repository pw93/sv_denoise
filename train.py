# -*- coding: utf-8 -*-

import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.unet import UNet
from model.dncnn import DnCNN

#from denoise_dataset import DenoiseDataset  # Your Dataset class for noisy ↔ clean pairs
from denoise_dataset_mem import DenoiseDataset_mem  # Your Dataset class for noisy ↔ clean pairs

from torchvision import transforms

from PIL import Image

#==============================

dname_noise = r'D:\data\dataset2\denoise50\train\input'
dname_gt = r'D:\data\dataset2\denoise50\train\gt'
num_epochs = 3000
batch_size = 32


#==============================

# Define your transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = DenoiseDataset_mem(
        noisy_dir=dname_noise,
        clean_dir=dname_gt,
        crop_size=(128, 128),
        transform=transform
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    #loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, persistent_workers=True, pin_memory=True)

    # Model, loss, optimizer
    #model = UNet(n_channels=3, n_classes=3).to(device)

    model = DnCNN(image_channels=3).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch}/{num_epochs}]  Loss: {avg_loss:.6f}  Time: {datetime.now()}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/unet_denoise_epoch{epoch}.pth")

import numpy as np

transform2 = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def denoise_image(model, image_path, device):
    """Denoise a single image."""
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)

    output_image = output.squeeze().cpu().numpy()

    print('==================')
    print(type(output_image))
    print(output_image.dtype)
    print(output_image.shape)

    #output_image = (output_image * 0.5) + 0.5
    '''
    output_image = (output_image * 255)
    output_image = np.clip(output_image,0,255)
    output_image = output_image.astype('uint8')
    output_image = Image.fromarray(output_image.transpose(1, 2, 0))
    '''

    output_image = (output_image * 255)
    output_image = np.clip(output_image,0,255).astype('uint8')
    output_image = Image.fromarray(output_image.transpose(1, 2, 0))

    return output_image

def process_images(input_dir, output_dir, model, device):
    """Process all images in the input directory."""
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            denoised_image = denoise_image(model, input_path, device)
            denoised_image.save(output_path)
            print(f"Saved denoised image to {output_path}")
            #break

def inference_train_image():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #!
    #model = UNet(n_channels=3, n_classes=3).to(device)
    model = DnCNN(image_channels=3).to(device)
    model.load_state_dict(torch.load('checkpoints/unet_denoise_epoch1000.pth', map_location=device))

    input_dir = r'D:\data\dataset2\denoise\test\input'
    output_dir = r'c:\temp3\unet_result'
    process_images(input_dir, output_dir, model, device)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    train()


    inference_train_image()


