# Machine Learning - Group 8

[Competition](https://www.kaggle.com/c/ugentml18-2)

## Group Members
  - [Freek Verschelden](mailto:Freek.Verschelden@UGent.be)
  - [Aron Minne](mailto:Aron.Minne@UGent.be)
  - [Simon Schellaert](mailto:Simon.Schellaert@UGent.be)
  - [Nico Ekkart](mailto:Nico.Ekkart@UGent.be)

## Tasks
  - Read about [pre-trained model](https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch) (F, A, S, N)
  - Read about [PyTorch](https://pytorch.org/docs/stable/index.html) (F, A, S, N)
  - Try pre-trained model (A)
  - Improve linear model (DAISY -> SURF) + merge other linear models (S)
  - Set up server on GCP and document process (F)
  - Try ensemble or something else (N)
  - Data augmentation

## Setup

Make sure you have the directory structure shown belown.

```
src
  train.py
  predict.py
  ...
data
  train
    bobcat
      bobcat_0001.jpg
      bobcat_0002.jpg
      ...
    chihuahua
      chihuahua_0001.jpg
      chihuahua_0002.jpg
      ...
    ...
  test
    unlabeled
      test_0001.jpg
      test_0002.jpg
      ...
```

Next, you can run `python3 train.py` to train the model or `python3 predict.py` to make predictions for the test set.
