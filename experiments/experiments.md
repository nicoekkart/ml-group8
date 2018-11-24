# Experiments 

## 1. ResNet 152 - All layers

### Setup
Training all layers of a pre-trained Resnet152 where the last fully-connected layer is replaced by `nn.Linear(model.fc.in_features, num_classes`).
No layers are frozen at any point during training.

### Train augmentations:
  - RandomResizedCrop(224, scale=(0.4, 1)),
  - RandomHorizontalFlip(),

### Hyperparameters:
  - CV Folds: 1
  - Learning rate: 0.001 (scheduled with step size 7 and gamma 0.1)
  - Batch size: 24
  - Epochs: 12
  - Splits: 5
  
### Results
Coming soon


# 2. ResNet 152 - FC

Identical setup to experiment 1 we only train the last FC layer.
All other layers are frozen

### Results
Coming soon


# Ideas
  - Increase/decrease learning rate
  - Increase/decrease batch size
  - Freeze convolutional layers during first part of training and then unfreeze them layer
  - Freeze first half of convolutional layers