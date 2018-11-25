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
```
Fold 0 - Epoch 0 - Train Loss: 0.7745
Fold 0 - Epoch 0 - Val Loss: 0.1621
Fold 0 - Epoch 1 - Train Loss: 0.1843
Fold 0 - Epoch 1 - Val Loss: 0.1185
Fold 0 - Epoch 2 - Train Loss: 0.1250
Fold 0 - Epoch 2 - Val Loss: 0.1069
Fold 0 - Epoch 3 - Train Loss: 0.0879
Fold 0 - Epoch 3 - Val Loss: 0.1016
Fold 0 - Epoch 4 - Train Loss: 0.0652
Fold 0 - Epoch 4 - Val Loss: 0.1134
Fold 0 - Epoch 5 - Train Loss: 0.0527
Fold 0 - Epoch 5 - Val Loss: 0.1133
Fold 0 - Epoch 6 - Train Loss: 0.0498
Fold 0 - Epoch 6 - Val Loss: 0.1018
Fold 0 - Epoch 7 - Train Loss: 0.0402
Fold 0 - Epoch 7 - Val Loss: 0.1063
Fold 0 - Epoch 8 - Train Loss: 0.0368
Fold 0 - Epoch 8 - Val Loss: 0.1063
Fold 0 - Epoch 9 - Train Loss: 0.0335
Fold 0 - Epoch 9 - Val Loss: 0.1085
Best loss per fold [0.10160743673279714]
Epoch for best loss per fold [3]
```



## 2. ResNet 152 - FC

Identical setup to experiment 1 we only train the last FC layer.
All other layers are frozen

### Results
```
Training last FC layer
Fold 0 - Epoch 0 - Train Loss: 0.9770
Fold 0 - Epoch 0 - Val Loss: 0.3399
Fold 0 - Epoch 1 - Train Loss: 0.3353
Fold 0 - Epoch 1 - Val Loss: 0.2075
Fold 0 - Epoch 2 - Train Loss: 0.2594
Fold 0 - Epoch 2 - Val Loss: 0.1744
Fold 0 - Epoch 3 - Train Loss: 0.2228
Fold 0 - Epoch 3 - Val Loss: 0.1588
Fold 0 - Epoch 4 - Train Loss: 0.2016
Fold 0 - Epoch 4 - Val Loss: 0.1473
Fold 0 - Epoch 5 - Train Loss: 0.1888
Fold 0 - Epoch 5 - Val Loss: 0.1321
Fold 0 - Epoch 6 - Train Loss: 0.1658
Fold 0 - Epoch 6 - Val Loss: 0.1298
Fold 0 - Epoch 7 - Train Loss: 0.1669
Fold 0 - Epoch 7 - Val Loss: 0.1301
Fold 0 - Epoch 8 - Train Loss: 0.1530
Fold 0 - Epoch 8 - Val Loss: 0.1267
Fold 0 - Epoch 9 - Train Loss: 0.1468
Fold 0 - Epoch 9 - Val Loss: 0.1235
Fold 0 - Epoch 10 - Train Loss: 0.1455
Fold 0 - Epoch 10 - Val Loss: 0.1220
Fold 0 - Epoch 11 - Train Loss: 0.1487
Fold 0 - Epoch 11 - Val Loss: 0.1220
```


## 3.
Identical setup to experiment 1 but with batch size 8 instead of 24.

### Results
```
Fold 0 - Epoch 0 - Train Loss: 0.5881
Fold 0 - Epoch 0 - Val Loss: 0.2265
Fold 0 - Epoch 1 - Train Loss: 0.2433
Fold 0 - Epoch 1 - Val Loss: 0.1614
Fold 0 - Epoch 2 - Train Loss: 0.1748
Fold 0 - Epoch 2 - Val Loss: 0.1742
Fold 0 - Epoch 3 - Train Loss: 0.1751
Fold 0 - Epoch 3 - Val Loss: 0.1536
Fold 0 - Epoch 4 - Train Loss: 0.1148
Fold 0 - Epoch 4 - Val Loss: 0.1432
Fold 0 - Epoch 5 - Train Loss: 0.0969
Fold 0 - Epoch 5 - Val Loss: 0.1587
Fold 0 - Epoch 6 - Train Loss: 0.1078
Fold 0 - Epoch 6 - Val Loss: 0.2166
Fold 0 - Epoch 7 - Train Loss: 0.0866
Fold 0 - Epoch 7 - Val Loss: 0.2012
Fold 0 - Epoch 8 - Train Loss: 0.0615
Fold 0 - Epoch 8 - Val Loss: 0.1742
Fold 0 - Epoch 9 - Train Loss: 0.0505
Fold 0 - Epoch 9 - Val Loss: 0.1622
Completed fold 0 with best loss of 0.14316712353336677
Best loss per fold [0.14316712353336677]
Mean: 0.14316712353336677
Stddev: 0.0
Epoch for best loss per fold [4]
```


## 4.
Identical setup to experiment 1 but with batch size 32 instead of 24.

### Results
```
Fold 0 - Epoch 0 - Train Loss: 0.8770
Fold 0 - Epoch 0 - Val Loss: 0.1884
Fold 0 - Epoch 1 - Train Loss: 0.1948
Fold 0 - Epoch 1 - Val Loss: 0.1241
Fold 0 - Epoch 2 - Train Loss: 0.1269
Fold 0 - Epoch 2 - Val Loss: 0.1070
Fold 0 - Epoch 3 - Train Loss: 0.0959
Fold 0 - Epoch 3 - Val Loss: 0.1075
Fold 0 - Epoch 4 - Train Loss: 0.0803
Fold 0 - Epoch 4 - Val Loss: 0.1044
Fold 0 - Epoch 5 - Train Loss: 0.0640
Fold 0 - Epoch 5 - Val Loss: 0.1107
Fold 0 - Epoch 6 - Train Loss: 0.0565
Fold 0 - Epoch 6 - Val Loss: 0.1106
Fold 0 - Epoch 7 - Train Loss: 0.0474
Fold 0 - Epoch 7 - Val Loss: 0.1099
Fold 0 - Epoch 8 - Train Loss: 0.0398
Fold 0 - Epoch 8 - Val Loss: 0.1087
Fold 0 - Epoch 9 - Train Loss: 0.0449
Fold 0 - Epoch 9 - Val Loss: 0.1121
Completed fold 0 with best loss of 0.10438271586814624
Best loss per fold [0.10438271586814624]
Mean: 0.10438271586814624
Stddev: 0.0
Epoch for best loss per fold [4]
```

# Ideas
  - Increase/decrease learning rate
  - Increase/decrease batch size
  - Freeze convolutional layers during first part of training and then unfreeze them layer
  - Freeze first half of convolutional layers
  - Try VGG19, DenseNet and Inception V3
 
