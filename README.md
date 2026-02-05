# Crop Classification with EfficientNet-B4 (End-to-End Deep Learning Pipeline)

This repository presents a **complete end-to-end deep learning workflow for agricultural crop image classification** using **PyTorch**.  
The project covers **data preprocessing, normalization, dataset splitting, model fine-tuning, training optimization, evaluation, class imbalance handling via undersampling, and retraining**.

The entire pipeline was initially developed in **Google Colab** with GPU acceleration and later prepared for GitHub version control.

---

## 1. Project Objective

The goal of this project is to:

- Build a **robust multi-class image classification model** for agricultural crops  
- Apply **transfer learning** using a pretrained **EfficientNet-B4** architecture  
- Perform **dataset-specific normalization**  
- Implement **training best practices** such as learning rate scheduling, early stopping, and checkpointing  
- Address **class imbalance** using undersampling  
- Evaluate performance both **quantitatively and visually**

---

## 2. Dataset Structure

The dataset follows the standard `ImageFolder` format required by `torchvision`:

- The dataset contains images of multiple crops stored in directories named after the crop classes. You can find this in data folder
data/
 ├── banana/
 ├── almond/
 ├── clove/
 └── ...
---

- Each subfolder corresponds to a **crop class**
- Images vary in size and color mode (RGB and non-RGB)

---

## 3. Environment & Dependencies

Key libraries used in this project include:

- `torch`, `torchvision`
- `torchinfo`
- `numpy`, `pandas`
- `matplotlib`
- `tqdm`
- `PIL`
- `scikit-learn`

GPU availability is automatically detected:

device = "cuda" if torch.cuda.is_available() else "cpu"

4. Data Preprocessing Pipeline
4.1 RGB Conversion

Some images were not stored in RGB format.
To ensure compatibility with EfficientNet, a custom transformation was implemented:

class ConvertToRGB(object):
    def __call__(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image


This guarantees all images have three channels.

4.2 Initial Image Transformations

Before normalization, images undergo the following preprocessing steps:

Conversion to RGB

Resize to 380 × 380 (EfficientNet-B4 input resolution)

Conversion to tensor

transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((380, 380)),
    transforms.ToTensor()
])

5. Dataset Statistics (Mean & Standard Deviation)

Instead of using ImageNet statistics, dataset-specific normalization was computed.

5.1 Mean and Standard Deviation Computation

A custom function iterates through the dataset to calculate:

Channel-wise mean

Channel-wise variance and standard deviation

mean, std = get_mean_std(data_loader)


This ensures normalization is tailored to the agricultural dataset, improving training stability.

5.2 Normalized Dataset

A normalized dataset is created using the computed statistics:

transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


A second pass verifies that the normalized dataset has approximately zero mean and unit variance.

6. Train / Validation Split

The dataset is split into training and validation subsets:

80% training

20% validation

Fixed random seed for reproducibility

g = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(norm_dataset, (0.8, 0.2), generator=g)

6.1 Class Distribution Analysis

Class counts are computed after splitting to verify class representation.
Bar charts are plotted for both training and validation datasets.

7. Data Loading Strategy

Separate data loaders are used:

Loader	Shuffle	Purpose
Train Loader	Yes	Model training
Validation Loader	No	Model evaluation

Batch size: 32

8. Model Architecture & Transfer Learning
8.1 Base Model

EfficientNet-B4

Pretrained on ImageNet

models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)

8.2 Freezing Pretrained Layers

All feature extractor layers are frozen to prevent overfitting:

for param in model.parameters():
    param.requires_grad = False

8.3 Custom Classification Head

The original classifier is replaced with:

Linear → ReLU → Dropout → Linear


Hidden units: 500

Dropout rate: 0.5

Output units: number of crop classes

This enables task-specific learning.

9. Training Strategy
9.1 Loss Function and Optimizer

Loss: CrossEntropyLoss

Optimizer: Adam

Learning rate: 0.001

Weight decay: 1e-4

9.2 Learning Rate Scheduler

A StepLR scheduler is used:

Step size: 4 epochs

Gamma: 0.2

This gradually reduces the learning rate during training.

9.3 Early Stopping

Training stops if validation loss does not improve for 5 consecutive epochs, preventing overfitting.

9.4 Model Checkpointing

The best model (based on validation loss) is saved automatically:

model/LR_model.pth


The checkpoint includes:

Model weights

Optimizer state

Best validation loss

10. Model Evaluation

Evaluation metrics include:

Training and validation loss

Training and validation accuracy

Learning rate progression

Confusion matrix

Per-image qualitative predictions

Metrics are stored in pandas DataFrames for analysis.

11. Prediction & Visualization
11.1 Probability Prediction

Softmax probabilities are computed on the validation dataset.

11.2 Confusion Matrix

A confusion matrix is generated using scikit-learn to visualize class-level performance.

11.3 Visual Inspection

Random validation images are displayed with their predicted crop labels to assess qualitative performance.

12. Class Imbalance Handling (Undersampling)

To address dataset imbalance:

A custom undersampling function was implemented

Each class is reduced to the same number of samples

Files are copied into a new directory

undersampled/
├── banana/
├── maize/
└── ...

12.1 Retraining on Balanced Dataset

The entire pipeline is repeated on the undersampled dataset:

Dataset loading

Train/validation split

Training

Evaluation

Confusion matrix

Prediction visualization

This enables direct comparison between:

Original imbalanced dataset

Balanced (undersampled) dataset

13. Key Takeaways

Dataset-specific normalization improves training stability

Transfer learning significantly reduces training time

Undersampling improves fairness across minority classes

Learning rate scheduling and early stopping reduce overfitting

Visual inspection complements numerical metrics
