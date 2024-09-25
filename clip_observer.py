import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from nih_chest_xray_reader import NIHChestXrayDataset

class CLIPModelObserver:
    def __init__(self, device=None, verbose=False, batch_size=32):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.labels = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        self.template = 'Chest Radiograph, Diagnosis: '
        self.verbose = verbose
        self.batch_size = batch_size
        
    def predict_batch(self, images):
        probabilities_batch = []

        for image in images:
            batch_probs = []

            with torch.no_grad():
                for label in self.labels:
                    text_prompts = [f"{self.template}{label}", f"{self.template}No {label}"]
                    
                    # Process the image and text with CLIP processor
                    inputs = self.processor(text=text_prompts, images=image, return_tensors="pt", padding=True).to(self.device)
                    outputs = self.model(**inputs)

                    # Calculate probabilities from logits
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                    batch_probs.append(probs[0][0].item())  # Probability for the disease

            probabilities_batch.append(batch_probs)

        return np.array(probabilities_batch)


    def evaluate(self, images, ground_truth_labels, max_fpr=0.1):

        # Convert ground truth to numpy if necessary
        if isinstance(ground_truth_labels, torch.Tensor):
            ground_truth_labels = ground_truth_labels.cpu().numpy()

        num_images = len(images)
        results = np.zeros((len(self.labels), 5))  # For TP, FP, FN, TN, AUC for each disease
        all_predictions = []

        for i in range(0, num_images, self.batch_size):
            if self.verbose:
                print(f"Processing batch {i // self.batch_size + 1}/{(num_images + self.batch_size - 1) // self.batch_size}...")

            batch_images = images[i:i + self.batch_size]
            probabilities_batch = self.predict_batch(batch_images)
            all_predictions.append(probabilities_batch)

            if self.verbose:
                for j in range(len(batch_images)):
                    print(f"File: {i+j+1}, True: {ground_truth_labels[i+j]}, Probabilities: {probabilities_batch[j]}")

        all_predictions = np.vstack(all_predictions)

        for j, label in enumerate(self.labels):
            y_true = ground_truth_labels[:, j]
            y_scores = all_predictions[:, j]

            # Compute AUC
            if len(np.unique(y_true)) < 2:
                if self.verbose:
                    print(f"Skipping AUC calculation for '{label}' due to insufficient data.")
                auc = np.nan
            else:
                auc = roc_auc_score(y_true, y_scores)
            
            results[j, 4] = auc

            # Determine threshold based on max FPR
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            threshold_idx = np.where(fpr <= max_fpr)[0][-1]  # Last index where FPR <= max_fpr
            threshold = thresholds[threshold_idx]

            # Apply the threshold to predictions
            predictions = (y_scores >= threshold).astype(int)

            # Calculate TP, FP, FN, TN
            results[j, 0] = np.sum((predictions == 1) & (y_true == 1))  # TP
            results[j, 1] = np.sum((predictions == 1) & (y_true == 0))  # FP
            results[j, 2] = np.sum((predictions == 0) & (y_true == 1))  # FN
            results[j, 3] = np.sum((predictions == 0) & (y_true == 0))  # TN

        return results, all_predictions

    def save_results(self, results, predictions, filename):
        np.savetxt(filename, results, fmt='%.4f', delimiter=',', header="TP,FP,FN,TN,AUC", comments='')
        np.save(filename.replace('.csv', '_probabilities.npy'), predictions)

    def print_evaluation(self, results, filename=None, predictions=None):
        if filename and predictions is not None:
            self.save_results(results, predictions, filename)
        
        print(f"{'Disease':<20}{'TP':<10}{'FP':<10}{'FN':<10}{'TN':<10}{'AUC':<10}")
        for i, label in enumerate(self.labels):
            tp, fp, fn, tn, auc = results[i]
            print(f"{label:<20}{int(tp):<10}{int(fp):<10}{int(fn):<10}{int(tn):<10}{auc:<10.4f}")

# Example usage
if __name__ == "__main__":
    # Assuming NIHChestXrayDataset is defined and working
    dataset = NIHChestXrayDataset(
        root_dir='../../data/NIH_Chest_Xray',
        csv_file='Data_Entry_2017.csv',
        image_folder_prefix='images_',
        max_folders=12,
        mode='test'  # Using test mode for evaluation
    )

    observer = CLIPModelObserver(verbose=True, batch_size=64)
    data_loader = DataLoader(dataset, batch_size=64, num_workers=16, shuffle=False)

    all_images = []
    all_labels = []

    num_images = 4096

    # Load the images in batches and accumulate
    for batch_images, batch_labels in data_loader:
        all_images.extend(batch_images)
        all_labels.extend(batch_labels)
        if len(all_images) >= num_images:
            break  # Stop when we have loaded 4096 images

    # Truncate to exactly 4096 images
    sample_images = all_images[:num_images]
    sample_labels = all_labels[:num_images]

    # Convert tensors to PIL images for the observer
    _sample_images = []
    for img in sample_images:
        img = img.squeeze().numpy()
        img = (img * 255).astype(np.uint8)
        _img = Image.fromarray(img, mode='L')
        _sample_images.append(_img)
    sample_images = _sample_images

    # Evaluate predictions
    results, predictions = observer.evaluate(sample_images, torch.stack(sample_labels))

    # Save and print the evaluation
    observer.print_evaluation(results, filename="clip_results.csv", predictions=predictions)
