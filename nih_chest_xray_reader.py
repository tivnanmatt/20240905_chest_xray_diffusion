import os
import pandas as pd
import random
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor

class MissingFileWarning(Exception):
    """Custom warning to be raised when an image file is not found after several attempts."""
    pass

class NIHChestXrayDataset(Dataset):
    LABELS = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
        "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
    ]

    def __init__(self, root_dir, csv_file, image_folder_prefix='images_', max_folders=12, transform=None, mode='train', verbose=False):
        self.root_dir = root_dir
        self.csv_file = os.path.join(root_dir, csv_file)
        self.image_folder_prefix = image_folder_prefix
        self.max_folders = max_folders
        self.data = pd.read_csv(self.csv_file)
        self.transform = transform if transform else self.default_transform()
        self.missing_file_count = 0
        self.verbose = verbose
        self.mode = mode

        # Load the train/val/test split lists
        self.image_list = self.load_image_list()

    def load_image_list(self):
        """Load the list of image file names for the specified mode (train, val, or test)."""
        list_file = 'test_list.txt' if self.mode == 'test' else 'train_val_list.txt'
        list_path = os.path.join(self.root_dir, list_file)
        
        with open(list_path, 'r') as f:
            filenames = f.read().splitlines()

        # Split the list for 'train' and 'val' modes
        total_files = len(filenames)
        train_split_index = int(0.8 * total_files)
        
        if self.mode == 'train':
            filenames = filenames[:train_split_index]
        elif self.mode == 'val':
            filenames = filenames[train_split_index:]
        
        return filenames

    def default_transform(self):
        """Define default image transformations (Resize, CenterCrop, ToTensor)."""
        return Compose([
            Resize(size=224, interpolation=Image.BICUBIC),
            CenterCrop(size=(224, 224)),
            ToTensor()  # Returns tensor with values normalized between [0, 1]
        ])

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_list)

    def __getitem__(self, idx):
        """Retrieve an image and its label based on the index."""
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            samples = [self.__getitem__(i) for i in indices]
            images, labels = zip(*samples)
            return torch.stack(images), torch.stack(labels)
        
        img_filename = self.image_list[idx]
        label = self.get_label_vector(img_filename)

        try:
            image = self._find_image_file(img_filename)
            self.missing_file_count = 0  # Reset missing file counter after successful load
            if self.verbose:
                print(f"Loaded image: {img_filename}")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            if self.verbose:
                print(f"Warning: Issue with the image file '{img_filename}'. Proceeding with a fallback case.")
            self.missing_file_count += 1
            if self.missing_file_count >= 5:
                raise MissingFileWarning(f"Multiple issues with files in sequence. Checked up to index {idx}. Please verify the dataset integrity.")
            
            # Fallback to the next highest index deterministically
            next_idx = (idx + 1) % len(self.image_list)
            return self.__getitem__(next_idx)
        
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

    def get_label_vector(self, img_filename):
        """Convert the label from the dataset into a multi-hot encoded vector."""
        row = self.data[self.data['Image Index'] == img_filename]
        labels = row['Finding Labels'].iloc[0].split('|')
        
        label_vector = [0] * len(self.LABELS)
        for label in labels:
            if label in self.LABELS:
                label_vector[self.LABELS.index(label)] = 1
        return label_vector

    def _find_image_file(self, img_filename):
        """Locate the image file in the incrementing folders (image_folder_prefix + '001', etc.)."""
        for i in range(1, self.max_folders + 1):
            folder_name = f"{self.image_folder_prefix}{i:03}/images"
            img_path = os.path.join(self.root_dir, folder_name, img_filename)
            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path)
                    image = image.convert('L')  # Convert the image to grayscale (1 channel)
                    return image
                except (UnidentifiedImageError, OSError, SyntaxError) as e:
                    if self.verbose:
                        print(f"Warning: Issue reading image file '{img_path}'. Error: {e}")

        raise FileNotFoundError(f"Image file {img_filename} not found or could not be read in any folder from {self.image_folder_prefix}001 to {self.image_folder_prefix}{self.max_folders:03}.")

    def sample(self, nBatch=1):
        """Randomly sample `nBatch` images and labels from the dataset."""
        indices = random.sample(range(len(self.image_list)), nBatch)
        samples = [self.__getitem__(idx) for idx in indices]
        images, labels = zip(*samples)
        return torch.stack(images), torch.stack(labels)

# Example usage
if __name__ == "__main__":
    dataset = NIHChestXrayDataset(
        root_dir='../../data/NIH_Chest_Xray',
        csv_file='Data_Entry_2017.csv',
        image_folder_prefix='images_',
        max_folders=12,
        mode='train',  # Can be 'train', 'val', or 'test'
        verbose=True  # Set to True to print warnings
    )

    # Get a random sample of images and labels
    random_sample, random_labels = dataset.sample(nBatch=256)
    print(random_sample.shape)  # Should print torch.Size([256, 1, 224, 224])
    print(random_labels.shape)  # Should print torch.Size([256, 14])
    print(random_labels)
    # Print the sum of the labels in each dimension
    print(torch.sum(random_labels, dim=0))
    print(torch.sum(random_labels, dim=1))  

    # Perform a deterministic sample
    deterministic_sample, deterministic_labels = dataset[0:7]
    print(deterministic_sample.shape)  # Should print torch.Size([7, 1, 224, 224])
    print(deterministic_labels.shape)  # Should print torch.Size([7, 14])
