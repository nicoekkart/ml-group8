import yaml
import random

sweep_configuration = {
    'feature_count': [1500, 3000, 5000],
    'codebook_size': [0.25, 0.5, 0.75, 1],
    'regularization': [0.5, 1, 2, 3]
}

combinations = [{}]

for key, values in sweep_configuration.items():
    combinations = [{key: value, **combination}  for combination in combinations for value in values]

random.shuffle(combinations)

for i, combination in enumerate(combinations):
    with open('config/' + str(i) + '.yml', 'w') as config_file:
        yaml.dump(combination, config_file)
