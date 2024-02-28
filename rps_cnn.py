import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import platform
import datetime
import os
import math
import random


# Check the Environment of Python, Tensorflow, Keras
print('Python version:', platform.python_version())
print('Tensorflow version:', tf.__version__)
print('Keras version:', tf.keras.__version__)

# Set the Dataset
DATASET_NAME = 'rock_paper_scissors'
(dataset_train_raw, dataset_test_raw), dataset_info = tfds.load(
    name=DATASET_NAME,
    data_dir='tmp',
    with_info=True,
    as_supervised=True,
    split=[tfds.Split.TRAIN, tfds.Split.TEST],
)
# Check the train&test dataset
print('Raw train set size:', len(list(dataset_train_raw)))
print('Raw test set size:', len(list(dataset_test_raw)))
print('Dataset Info', dataset_info)

NUM_TRAIN_EXAMPLES = dataset_info.splits['train'].num_examples
NUM_TEST_EXAMPLES = dataset_info.splits['test'].num_examples
NUM_CLASSES = dataset_info.features['label'].num_classes

print('Number of training examples:', NUM_TRAIN_EXAMPLES)
print('Number of test examples:', NUM_TEST_EXAMPLES)
print('Number of labeled classes:', NUM_CLASSES)