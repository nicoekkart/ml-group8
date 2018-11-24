# Experiments 

## ResNet 152 - FC

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
