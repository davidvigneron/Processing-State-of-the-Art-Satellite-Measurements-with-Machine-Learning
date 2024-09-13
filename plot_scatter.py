import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mpl_scatter_density # adds projection='scatter_density'
import numpy as np
import sys

white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (sys.float_info.epsilon, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=512)


def plot_scatter(predictions, truth, var_name, test_score):
    """
    This function was used to create the scatter plots predictions over truth.

    predictions and truth are the respective arrays for all observations and are expected to have the same shape.
    var_name is the name of the variable for the current array, this is added to the title(s)
    test_score contains r2 scores for every level
    
    """
    fig, axes = plt.subplots(5, 6, figsize=(15, 12), subplot_kw={'projection': 'scatter_density'})  # 5 rows and 6 columns of subplots
    fig.suptitle(f'Comparison of predicted and true values for {var_name}')
    for i in range(truth.shape[1]):
        row, col = divmod(i, 6)
        ax = axes[row, col]
        cb = ax.scatter_density(predictions[:,i], truth[:,i], cmap=white_viridis, dpi=600)
        plt.colorbar(cb, ax=ax)
        mean = np.mean(truth[:,i])
        ax.axline((mean, mean), slope=1, c="black", linestyle="--", linewidth=.5, alpha=.5)
        ax.set_title(f'{var_name} lvl {i}\nR2: {test_score[i]:<.6}', fontsize='x-small')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Full physics")
    # Hide any unused subplots
    for i in range(truth.shape[1], 30):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout(rect=(0, 0, 0.98, 1))
    return fig