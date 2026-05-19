import yaml
import torch

from ucca.model import UCCA
from data import get_data, prepare_data
from utils import get_total_correlation

def main():
    n_test = 256
    output_size = 2
    
    print("Loading data...")
    # Load data
    A, B, labels = get_data('handwritten', view1_idx=2, view2_idx=3)
    
    print("Preparing data...")
    # Prepare data (train/test split, center)
    X_train, X_test, Y_train, Y_test, L_X_train, L_Y_train, L_test = prepare_data(
        A, B, n_test=n_test, split='unpaired', labels=labels
    )
    
    # Run UCCA fit and transform
    print("Running UCCA fit...")
    model = UCCA(
        output_size=output_size,
        anchors_n_runs=10,
        pca_n_components=10,
    )
    model.fit(X_train, Y_train)
    
    print("Running UCCA transform on test set...")
    X_output, Y_output = model.transform(X_test, Y_test)
    
    print("Done! UCCA execution successful.")
    print("Shape of X_output:", X_output.shape)
    print("Shape of Y_output:", Y_output.shape)
    
    tc = get_total_correlation(X_output, Y_output)
    print(f"Total Correlation (TC): {tc:.4f}")

if __name__ == '__main__':
    main()
