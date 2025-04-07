import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(data: np.ndarray):
    """
    Plots a histogram of a NumPy array containing integers.
    The number of bins is determined automatically to match the distribution.

    Parameters:
        data (np.ndarray): A NumPy array of integers.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if data.dtype.kind not in {'i', 'u'}:
        raise ValueError("Array must contain integers.")
    
    # Use np.histogram_bin_edges with 'auto' to find optimal bins
    bins = np.histogram_bin_edges(data, bins='auto')
    
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.75)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Integer Data")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()



def plot_continuous(data: np.ndarray):
    """
    Plots a histogram of a NumPy array containing continuous (non-integer) values.
    The number of bins is determined automatically to match the distribution.

    Parameters:
        data (np.ndarray): A NumPy array of continuous values.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if data.dtype.kind not in {'f'}:
        raise ValueError("Array must contain continuous (float) values.")
    
    # Use np.histogram_bin_edges with 'auto' to find optimal bins
    bins = np.histogram_bin_edges(data, bins='auto')
    
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.75)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Continuous Data")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()