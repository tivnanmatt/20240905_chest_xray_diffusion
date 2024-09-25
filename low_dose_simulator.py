import torch
import random
import matplotlib.pyplot as plt

import laboratory as lab
from laboratory.torch.distributions.poisson import ConditionalPoissonDistribution

from nih_chest_xray_reader import NIHChestXrayDataset

import numpy as np

# class LowDoseChestXraySimulator(lab.torch.distributions.poisson.ConditionalPoissonDistribution):
class LowDoseChestXraySimulator(ConditionalPoissonDistribution):
    def __init__(self, I0=1e3, scale=5.0):
        # Convert to torch tensors and handle scalars as batch size 1
        self.I0 = I0 if torch.is_tensor(I0) else torch.tensor([I0], dtype=torch.float32)
        self.scale = scale if torch.is_tensor(scale) else torch.tensor([scale], dtype=torch.float32)
        super(LowDoseChestXraySimulator, self).__init__(self.gamma_fn)

    def _ensure_batch_size(self, tensor, batch_size):
        """Ensure the tensor has the required batch size."""
        if tensor.shape[0] == 1:  # If tensor has batch size 1, repeat it to match batch_size
            tensor = tensor.repeat(batch_size, *[1 for _ in tensor.shape[1:]])
        return tensor

    def gamma_fn(self, l):
        # Ensure both I0 and scale are tensors and check their shapes
        I0 = self.I0
        scale = self.scale

        # If I0 or scale are scalars or have batch size 1, repeat them to match the batch size of l
        if I0.dim() == 1:
            I0 = I0.view(-1, 1)  # Reshape if needed
        if scale.dim() == 1:
            scale = scale.view(-1, 1)

        batch_size = l.shape[0]

        # Ensure batch size consistency by repeating I0 or scale if necessary
        I0 = self._ensure_batch_size(I0, batch_size)
        scale = self._ensure_batch_size(scale, batch_size)

        # Expand I0 and scale to match the spatial dimensions of l
        I0 = I0.view(batch_size, 1, 1, 1).expand_as(l)
        scale = scale.view(batch_size, 1, 1, 1).expand_as(l)

        # Calculate gamma function output
        y_bar = I0 * torch.exp(-scale * l)
        return y_bar
    
    def estimate_line_integrals(self, y):
        I0 = self.I0
        scale = self.scale
        
        # Reshape if necessary
        if I0.dim() == 1:
            I0 = I0.view(-1, 1)
        if scale.dim() == 1:
            scale = scale.view(-1, 1)

        # Ensure batch size consistency
        batch_size = y.shape[0]
        I0 = self._ensure_batch_size(I0, batch_size)
        scale = self._ensure_batch_size(scale, batch_size)
        
        # Expand I0 and scale to match the spatial dimensions of y
        I0 = I0.view(batch_size, 1, 1, 1).expand_as(y)
        scale = scale.view(batch_size, 1, 1, 1).expand_as(y)

        # Clamp the values and estimate line integrals
        y_min = I0 * torch.exp(-scale * torch.ones_like(y))
        y = torch.clamp(y, min=y_min)
        return -torch.log(y / I0) / scale
    
    def sample_line_integrals(self, l):
        # Ensure the batch size of I0 and scale matches that of l before sampling
        batch_size = l.shape[0]
        self.I0 = self._ensure_batch_size(self.I0, batch_size)
        self.scale = self._ensure_batch_size(self.scale, batch_size)

        # Sample y and estimate line integrals
        y = self.sample(l)
        l_hat = self.estimate_line_integrals(y)
        return l_hat
    
# Main script
if __name__ == "__main__":
    # Load the dataset
    dataset = NIHChestXrayDataset(
        root_dir='../../data/NIH_Chest_Xray',
        csv_file='Data_Entry_2017.csv',
        image_folder_prefix='images_',
        max_folders=12,
        mode='train',
        verbose=True
    )

    # Instantiate the low-dose simulator with a mix of batch and scalar inputs
    I0 = torch.tensor([[200.0]])  # Example for a batch of I0 values
    scale = 5.0  # Example with a scalar scale value
    simulator = LowDoseChestXraySimulator(I0=I0, scale=scale)

    # Get a random image from the dataset
    random_index = random.randint(0, len(dataset) - 1)
    original_image, _ = dataset[random_index]

    # Handle batch size: if original_image has batch size 1, repeat it to match I0's batch size
    if original_image.dim() == 3:
        original_image = original_image.unsqueeze(0)  # Add a batch dimension if missing
    if I0.shape[0] > 1 and original_image.shape[0] == 1:
        original_image = original_image.repeat(I0.shape[0], 1, 1, 1)

    # Simulate low-dose measurements
    low_dose_image = simulator.sample_line_integrals(original_image)

    # Convert images to numpy for plotting
    original_image_np = original_image[0].squeeze().numpy()  # Plot only the first image
    low_dose_image_np = low_dose_image[0].squeeze().numpy()

    # Plot original and low-dose images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original image
    im1 = axes[0].imshow(original_image_np, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Original X-ray")
    fig.colorbar(im1, ax=axes[0])

    # Low-dose image
    im2 = axes[1].imshow(low_dose_image_np, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Low Dose X-ray")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig("low_dose_simulation.png")
