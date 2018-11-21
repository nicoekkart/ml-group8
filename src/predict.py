import argparse
import torch
import torch.nn as nn

from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from torch.utils import data

import csv


def write_predictions_csv(predictions, out_path, label_strings):
    """ Writes the predictions to a csv file. """
    with open(out_path, 'w') as outfile:
    # Initialise the writer
        csvwriter = csv.writer(
            outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # Write the header
        row_to_write=['Id'] + [label for label in label_strings]
        csvwriter.writerow(row_to_write)
        # Write the rows using 18 digit precision
        for idx, prediction in enumerate(predictions):
          assert len(prediction) == len(label_strings)
          csvwriter.writerow(
              [str(idx+1)] +
              ["%.18f" % p for p in prediction])

if __name__ == '__main__':
    # TODO: Don't hardcode this
    classes = ['bobcat', 'chihuahua', 'collie', 'dalmatian', 'german_shepherd', 'leopard', 'lion', 'persian_cat', 'siamese_cat', 'tiger', 'wolf']

    # TODO: Allow passing model and data set as parameters
    parser = argparse.ArgumentParser(description='Model Training Script')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Split the full dataset in a training and validation set
    test_loader = data.DataLoader(datasets.ImageFolder('../data/test', transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])), batch_size=24, shuffle=False)

    # TODO: Load model
    model = models.resnet152(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    model.load_state_dict(torch.load('0.torch', map_location=device))

    model = model.to(device)

    model.eval()

    softmax = nn.Softmax()
    predictions = []

    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            outputs = model(inputs.to(device))
            probabilities = softmax(outputs)

            for j in range(probabilities.shape[0]):
                predictions.append(list(probabilities[j].numpy()))

    # TODO: Generate random output file name
    write_predictions_csv(predictions, 'testpred', classes)




