# flowers_classifier
Training EfficientNet-B0 from scratch for 104-class flower classification using PyTorch. Focus on building a robust CV pipeline, label smoothing, and Macro F1-score optimization without Transfer Learning

# Flower Classification with EfficientNet-B0

Multi-class image classification of 104 flower species using EfficientNet-B0 trained from scratch (without pre-trained weights) in PyTorch.

The project focuses on building a complete computer vision pipeline and understanding neural network training mechanics, rather than relying on transfer learning.

## Dataset

- **Classes:** 104
- **Train:** 12,753 images
- **Validation:** 3,712 images
- **Test:** 7,382 images
- **Resolution:** 224x224
- **Source:** Kaggle (TFRecord format, converted to ImageFolder structure)

## Model Architecture

- **Base:** `torchvision.models.efficientnet_b0` (random initialization)
- **Classifier head:** `nn.Linear(1280, 104)`
- **Total parameters:** 4.14M
- **Input:** 3x224x224

## Training Details

| Parameter | Value |
|-----------|-------|
| Epochs | 80 |
| Batch size | 16 |
| Learning rate | 1e-3 |
| Optimizer | AdamW (weight decay 0.01) |
| Loss | CrossEntropyLoss (label smoothing 0.1) |
| Scheduler | ReduceLROnPlateau (mode=max, factor=0.5, patience=5) |
| Metric | Macro F1-score |

### Data Augmentation (train)

- Random crop, horizontal/vertical flip
- Random rotation (25°)
- Color jitter (brightness, contrast, saturation, hue)
- Random affine (translate, scale)
- Normalization with ImageNet statistics

## Results

- **Best validation Macro F1-score:** 0.69
- The result confirms that the training pipeline is correctly implemented and the model converges when trained from scratch.

## Project Structure
