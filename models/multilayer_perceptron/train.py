
import sys
from pathlib import Path

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

import pandas as pd
import numpy as np

from multilayer_perceptron import MultilayerPerceptron



DATA_DIR = Path("data/mnist/")
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

def process_csv(dataset, label_col_idx, has_labels=True):
    """
    Helper function to streamline preprocessing csv data
    """
    data = pd.read_csv(dataset, skip_blank_lines=False, header=0)
    labels = np.empty([data.shape[0], 1])
    features = data.iloc[:, 1:]

    if has_labels:
        labels = data.iloc[:, label_col_idx]

    return features, labels


train_X, train_y = process_csv(str(DATA_DIR / "mnist_train.csv"), 0)
print(f"Shape of training features: {train_X.shape} | Shape of corresponding labels: {train_y.shape}\n")
test_X, test_y = process_csv(str(DATA_DIR / "mnist_test.csv"), 0)
print(f"Shape of testing features: {test_X.shape} | Shape of corresponding labels: {test_y.shape}")


mlp_model = MultilayerPerceptron(
    features = train_X,
    labels = train_y,
    layers = [784, 256, 256, 128, 10],
    epsilon = 1e-4
)

print(f"""Starting model training with:
        num_epochs: {NUM_EPOCHS}
        learning_rate: {LEARNING_RATE}""")

train_history = mlp_model.train(
    epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE
)

