import json
import sys

import data.feta_dataset as feta_dataset

with open(sys.argv[1], 'r') as f:
    config = json.load(f)

data, labels = feta_dataset.get_path(config['data_path'])
dataset = feta_dataset.FeTA(
    data, 
    labels
)
import IPython; IPython.embed(); exit(1)