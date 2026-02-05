# Crop Classification using EfficientNetB4

## Project Overview
This project focuses on classifying various agricultural crops from image data using a deep learning approach. The goal is to build a robust model capable of accurately identifying different crop types, which can be beneficial for agricultural monitoring, yield prediction, and disease detection.

## Dataset
The dataset used for this project consists of images of 30 different agricultural crop types, sourced from the `Agricultural-crops` directory. 

Initially, the dataset exhibited class imbalance, which is a common challenge in real-world image datasets. To address this, an undersampling strategy was implemented to create a more balanced subset for training, ensuring that the model does not become biased towards majority classes.

## Methodology

### 1. Data Preprocessing
*   **Image Transformation**: Images were converted to RGB format, resized to `(380, 380)`, and converted to PyTorch tensors.
*   **Normalization**: The dataset's mean and standard deviation were calculated (mean: `[0.4782, 0.5156, 0.3332]`, std: `[0.2537, 0.2383, 0.2710]`) and applied for normalization, which helps in stabilizing and accelerating the training process.
*   **Data Splitting**: The dataset was split into training (80%) and validation (20%) sets. For the undersampled dataset, a similar split was performed.

### 2. Model Architecture
*   **Base Model**: `EfficientNetB4` was chosen as the base model, pretrained on ImageNet, leveraging the power of transfer learning.
*   **Feature Extraction**: The pre-trained `EfficientNetB4` layers (`model.features`) were frozen to retain their learned feature extraction capabilities.
*   **Custom Classifier Head**: A new classification head (`nn.Sequential`) was added on top of the frozen base model, consisting of:
    *   A `Linear` layer with input features matching the `EfficientNetB4`'s classifier input (`1792`).
    *   `ReLU` activation function.
    *   `Dropout` layer with `p=0.5` for regularization.
    *   A final `Linear` layer mapping to the number of output classes (30).

### 3. Training Strategy
*   **Loss Function**: `nn.CrossEntropyLoss` was used, suitable for multi-class classification.
*   **Optimizer**: `Adam` optimizer with a learning rate of `0.001` and `weight_decay=1e-4`.
*   **Learning Rate Scheduler**: `StepLR` was employed to reduce the learning rate by a factor of `gamma=0.2` every `step_size=4` epochs.
*   **Early Stopping**: Training was halted if the validation loss did not improve for 5 consecutive epochs to prevent overfitting.
*   **Checkpointing**: The model state with the best validation loss was saved.

### 4. Undersampling
To address class imbalance, an `undersample_dataset` function was implemented. This function samples a `target_count` (which is the minimum count of any class in the original dataset) number of images from each class, creating a new, more balanced dataset. The model was then fine-tuned on this undersampled dataset.

## Results

### Original Dataset Training
The model was initially trained on the full, albeit imbalanced, dataset.
*   **Epochs**: The training ran for 19 epochs before early stopping was triggered.
*   **Final Training Loss**: ~0.2002
*   **Final Training Accuracy**: ~97.32%
*   **Final Validation Loss**: ~0.5985
*   **Final Validation Accuracy**: ~70.31%

### Undersampled Dataset Training
After undersampling to balance the class distribution, the model was further trained.
*   **Epochs**: The training ran for 16 epochs before early stopping was triggered.
*   **Final Training Loss**: ~0.3106
*   **Final Training Accuracy**: ~93.94%
*   **Final Validation Loss**: ~0.3177
*   **Final Validation Accuracy**: ~96.09%

The undersampling approach significantly improved validation accuracy, indicating a better generalization across classes.

## Visualizations
*   **Loss and Accuracy Plots**: Graphs showing training and validation loss/accuracy over epochs.
*   **Learning Rate Schedule**: Plot illustrating the dynamic changes in the learning rate during training.
*   **Confusion Matrix**: Visual representation of the model's performance on the validation set, detailing correct and incorrect classifications for each class. This was generated for both original and undersampled validation sets.
*   **Sample Predictions**: Displays sample images from the validation set with their corresponding predicted class labels.

## Conclusion
This project successfully demonstrates the application of transfer learning with EfficientNetB4 for crop classification. The implementation of an undersampling strategy proved crucial in improving the model's performance on the validation set, especially in handling class imbalance. The model achieved a validation accuracy of over 96% on the undersampled dataset, indicating its strong capability for the task.

## Future Work
*   Explore other data augmentation techniques to further enhance model generalization.
*   Experiment with different transfer learning models (e.g., ResNet, VGG) and fine-tuning strategies.
*   Investigate advanced sampling techniques (e.g., SMOTE) or weighted loss functions to handle imbalance.
*   Deploy the model as a web service for real-time crop classification.
