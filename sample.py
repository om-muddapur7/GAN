import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, ConcatDataset

# ✅ Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images
    transforms.ToTensor(),        # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for RGB
])

# ✅ Dataset Paths (Check & Load)
# dataset_dirs = ["/content/drive/MyDrive/2011_09_26_drive_0001_extract/image_00", "/content/drive/MyDrive/2011_09_26_drive_0001_extract/image_01", "/content/drive/MyDrive/2011_09_26_drive_0001_extract/image_02", "/content/drive/MyDrive/2011_09_26_drive_0001_extract/image_03", "/content/drive/MyDrive/2011_09_26_drive_0020_sync/2011_09_26/2011_09_26_drive_0020_sync/image_00", "/content/drive/MyDrive/2011_09_26_drive_0020_sync/2011_09_26/2011_09_26_drive_0020_sync/image_01", "/content/drive/MyDrive/2011_09_26_drive_0020_sync/2011_09_26/2011_09_26_drive_0020_sync/image_02", "/content/drive/MyDrive/2011_09_26_drive_0020_sync/2011_09_26/2011_09_26_drive_0020_sync/image_03", "/content/drive/MyDrive/2011_09_26_drive_0027_sync/2011_09_26/2011_09_26_drive_0027_sync/image_00", "/content/drive/MyDrive/2011_09_26_drive_0027_sync/2011_09_26/2011_09_26_drive_0027_sync/image_01", "/content/drive/MyDrive/2011_09_26_drive_0027_sync/2011_09_26/2011_09_26_drive_0027_sync/image_02", "/content/drive/MyDrive/2011_09_26_drive_0027_sync/2011_09_26/2011_09_26_drive_0027_sync/image_03", "/content/drive/MyDrive/2011_09_26_drive_0119_extract/2011_09_26/2011_09_26_drive_0119_extract/image_00", "/content/drive/MyDrive/2011_09_26_drive_0119_extract/2011_09_26/2011_09_26_drive_0119_extract/image_01", "/content/drive/MyDrive/2011_09_26_drive_0119_extract/2011_09_26/2011_09_26_drive_0119_extract/image_02", "/content/drive/MyDrive/2011_09_26_drive_0119_extract/2011_09_26/2011_09_26_drive_0119_extract/image_03", "/content/drive/MyDrive/2011_09_28_drive_0021_sync/2011_09_28/2011_09_28_drive_0021_sync/image_00", "/content/drive/MyDrive/2011_09_28_drive_0021_sync/2011_09_28/2011_09_28_drive_0021_sync/image_01", "/content/drive/MyDrive/2011_09_28_drive_0021_sync/2011_09_28/2011_09_28_drive_0021_sync/image_02", "/content/drive/MyDrive/2011_09_28_drive_0021_sync/2011_09_28/2011_09_28_drive_0021_sync/image_03", "/content/drive/MyDrive/2011_09_28_drive_0054_sync/2011_09_28/2011_09_28_drive_0054_sync/image_00", "/content/drive/MyDrive/2011_09_28_drive_0054_sync/2011_09_28/2011_09_28_drive_0054_sync/image_01", "/content/drive/MyDrive/2011_09_28_drive_0054_sync/2011_09_28/2011_09_28_drive_0054_sync/image_02", "/content/drive/MyDrive/2011_09_28_drive_0054_sync/2011_09_28/2011_09_28_drive_0054_sync/image_03" ]  # Add your dataset folder paths here
dataset_dirs = [r"D:\NLP\2011_09_26_drive_0027_sync\2011_09_26\2011_09_26_drive_0027_sync\image_00"]

# ✅ Load datasets from multiple folders
datasets_list = [datasets.ImageFolder(root=folder, transform=transform) for folder in dataset_dirs if os.path.exists(folder)]
full_dataset = ConcatDataset(datasets_list)  # Merge all datasets

# ✅ DataLoader
batch_size = 64
dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

print(f"Total images loaded: {len(full_dataset)}")  # Check dataset size

# ✅ Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# ✅ Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1)  # 🟢 Fix: Flatten correctly

# ✅ Initialize Models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# ✅ Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# ✅ Loss Function
criterion = nn.BCELoss()

# ✅ Training Loop
num_epochs = 100
fixed_noise = torch.randn(25, 100, 1, 1, device=device)  # Fixed noise for consistent visualization

for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)  # Dynamically update batch size

        # 🎯 Train Discriminator
        real_labels = torch.ones(batch_size, device=device)  # 🟢 Fix shape
        fake_labels = torch.zeros(batch_size, device=device) # 🟢 Fix shape
        
        optimizer_D.zero_grad()
        output_real = discriminator(real_images)
        loss_real = criterion(output_real, real_labels)

        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake_images = generator(noise)
        output_fake = discriminator(fake_images.detach())
        loss_fake = criterion(output_fake, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # 🎯 Train Generator
        optimizer_G.zero_grad()
        output_fake = discriminator(fake_images)
        loss_G = criterion(output_fake, real_labels)

        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1}/{num_epochs} | Loss_D: {loss_D.item()} | Loss_G: {loss_G.item()}")

    # ✅ Save Generated Images Every 10 Epochs
    if epoch % 10 == 0:
        with torch.no_grad():
            fake_samples = generator(fixed_noise).cpu()
        vutils.save_image(fake_samples, fr"D:\NLP\outputs_{epoch}.png", normalize=True)

# ✅ Generate and Visualize Final Results
with torch.no_grad():
    fake_samples = generator(fixed_noise).cpu()

fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for i, img in enumerate(fake_samples[:5]):
    axs[i].imshow(img.permute(1, 2, 0).numpy() * 0.5 + 0.5)
    axs[i].axis('off')
plt.show()
