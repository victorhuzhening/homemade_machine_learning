import numpy as np
import pandas as pd

def process_csv(dataset, label_col_idx, has_labels=True):
    """
    Helper function to streamline preprocessing csv data.

    Returns:
      
    """

    data = pd.read_csv(dataset, skip_blank_lines=False, header=0)
    labels = np.empty([data.shape[0], 1])
    features = data.iloc[:, 1:]

    if has_labels:
        labels = data.iloc[:, label_col_idx]

    
    return features, labels