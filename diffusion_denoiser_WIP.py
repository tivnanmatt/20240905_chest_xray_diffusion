import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from nih_chest_xray_reader import NIHChestXrayDataset
from low_dose_simulator import LowDoseChestXraySimulator
from laboratory.torch.networks import DenseNet, DiffusersUnet2D

# Define a diffusion model that takes the line integral reconstructions as conditional information
class DoseAwareConditionalDiffusionModel(torch.nn.Module):
    def __init__(self):
        super(DoseAwareConditionalDiffusionModel, self).__init__()

        # 1->16 fully connected dose encoder
        self.dose_encoder = DenseNet(input_shape=1, output_shape=16,
                                     hidden_channels_list=[256, 256, 256],
                                     activation='prelu')

        self.unet = DiffusersUnet2D(
            input_channels=18,  # x_t (1) + l_hat (1) + dose encoding (16)
            time_encoder_hidden_size=256,
            image_size=224,
            unet_in_channels=64,
            unet_base_channels=64,
            unet_out_channels=1,
            conditional_channels=0
        )

    def forward(self, I0, t, l_hat, x_t):
        # Encode the dose (I0)
        enc_dose = self.dose_encoder(I0)
        enc_dose = enc_dose.view(-1, enc_dose.size(1), 1, 1)
        enc_dose = enc_dose.expand(-1, -1, x_t.size(2), x_t.size(3))

        # Concatenate x_t, l_hat, and dose encoding
        z = torch.cat([x_t, l_hat, enc_dose], dim=1)
        x_0_hat = self.unet(z, t)
        return x_0_hat


# Function to sample time uniformly between 0 and 1
def sample_time_uniform(batch_size):
    t = torch.rand(batch_size, 1)
    return t

# Function to sample I0 log-uniformly between 1 and 1e5
def sample_I0_log_uniform(batch_size):
    s = torch.rand(batch_size,1)
    return torch.pow(10,  s* 5)

# Training function
def train_diffusion_model(train_dataset, val_dataset=None, num_epochs=5, num_iterations_train=100, num_iterations_val=10, batch_size=32):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DoseAwareConditionalDiffusionModel().to(device)

    # Define optimizer and loss function (MSE for denoising)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    mse_loss = torch.nn.MSELoss()

    # Instantiate the low-dose simulator
    simulator = LowDoseChestXraySimulator(scale=torch.tensor([5.0]).to(device))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16) if val_dataset else None

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_loader_iter = iter(train_loader)

        for _ in tqdm(range(num_iterations_train)):
            try:
                images, _ = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                images, _ = next(train_loader_iter)

            images = images.to(device)

            # Sample time uniformly between 0 and 1
            t = sample_time_uniform(images.size(0)).to(device)

            # Sample I0 log-uniformly between 1 and 1e5
            I0 = sample_I0_log_uniform(images.size(0)).to(device)
            
            # Simulate low-dose measurements
            simulator.I0 = I0
            l_hat = simulator.sample_line_integrals(images).to(device)

            # Simulate noisy image
            x_t = images + torch.randn_like(images) * torch.sqrt(t).view(-1, 1, 1, 1).to(device)

            # Forward pass
            optimizer.zero_grad()
            x_0_hat = model(I0, t, l_hat, x_t)

            # Compute loss (MSE between denoised image and ground truth)
            loss = mse_loss(x_0_hat, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Training Loss: {running_loss / num_iterations_train}")

        # Optionally validate on the validation dataset
        if val_loader:
            val_loss = validate_diffusion_model(model, val_loader, simulator, num_iterations_val, device)
            print(f"Validation Loss after Epoch {epoch + 1}: {val_loss}")

        # Save model weights after each epoch
        torch.save(model.state_dict(), 'diffusion_model_weights.pth')

def validate_diffusion_model(model, val_loader, simulator, num_iterations_val, device):
    mse_loss = torch.nn.MSELoss()
    model.eval()

    total_loss = 0.0
    val_loader_iter = iter(val_loader)

    with torch.no_grad():
        for _ in tqdm(range(num_iterations_val)):
            try:
                images, _ = next(val_loader_iter)
            except StopIteration:
                break

            images = images.to(device)

            # Simulate low-dose measurements
            l_hat = simulator.sample_line_integrals(images).to(device)

            # Sample time uniformly between 0 and 1
            t = sample_time_uniform(images.size(0)).to(device)

            # Sample I0 log-uniformly between 1 and 1e5
            I0 = sample_I0_log_uniform(images.size(0)).to(device)

            # Simulate noisy image
            x_t = images + torch.randn_like(images) * torch.sqrt(t).view(-1, 1, 1, 1).to(device)

            # Forward pass
            x_0_hat = model(I0, t, l_hat, x_t)

            # Compute loss
            loss = mse_loss(x_0_hat, images)
            total_loss += loss.item()

    avg_loss = total_loss / num_iterations_val
    return avg_loss


# Example usage
if __name__ == "__main__":
    train_dataset = NIHChestXrayDataset(
        root_dir='../../data/NIH_Chest_Xray',
        csv_file='Data_Entry_2017.csv',
        image_folder_prefix='images_',
        max_folders=12,
        mode='train',
        verbose=True
    )

    val_dataset = NIHChestXrayDataset(
        root_dir='../../data/NIH_Chest_Xray',
        csv_file='Data_Entry_2017.csv',
        image_folder_prefix='images_',
        max_folders=12,
        mode='val',
        verbose=True
    )

    train_diffusion_model(train_dataset, val_dataset=val_dataset, num_epochs=20, num_iterations_train=100, num_iterations_val=10, batch_size=32)
