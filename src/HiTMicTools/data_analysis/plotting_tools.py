import pandas as pd
from typing import List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.base import BaseEstimator


def plot_confusion_matrix(
    model: BaseEstimator,
    X: Union[np.ndarray, pd.DataFrame],
    true_y: np.ndarray,
    figsize: Tuple[int, int] = (8, 6),
    plot_figure: bool = True,
) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
    """
    Plot a confusion matrix for the given model and data.

    Args:
        model (BaseEstimator): The trained model.
        X (Union[np.ndarray, pd.DataFrame]): The input data to make predictions on.
        true_y (np.ndarray): The true labels for the input data.
        figsize (Tuple[int, int], optional): The figure size. Default is (8, 6).
        plot_figure (bool, optional): Whether to display the plot or return the figure and axis. Default is True.

    Returns:
        Union[None, Tuple[plt.Figure, plt.Axes]]: If plot_figure is True, displays the confusion matrix plot and returns None.
                                                  If plot_figure is False, returns a tuple of (plt.Figure, plt.Axes).
    """
    # Make predictions on the testing set
    y_pred = model.predict(X)

    print(classification_report(true_y, y_pred))

    # Create the confusion matrix
    cm = confusion_matrix(true_y, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        ax=ax,
        xticklabels=np.unique(true_y),
        yticklabels=np.unique(true_y),
        cbar_kws={"label": "Relative Frequency"},
    )

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.tick_params(axis="both", rotation=45)

    if plot_figure:
        plt.tight_layout()
        plt.show()
    else:
        return fig, ax


def explore_clusters(
    data: pd.DataFrame,
    algorithm: str,
    param_list: List[Union[Tuple[float, int], int]],
    figsize: Tuple[int, int] = (12, 12),
    rows_cols: Tuple[int, int] = (2, 2),
    plot_figure: bool = True,
) -> Union[None, Tuple[plt.Figure, np.ndarray]]:
    """
    Explore and visualize clusters using the specified clustering algorithm and parameter list.

    Args:
        data (pd.DataFrame): The input data containing 'UMAP1' and 'UMAP2' columns for visualization.
        algorithm (str): The clustering algorithm to use. Supported values are 'DBSCAN' and 'KMeans'.
        param_list (List[Union[Tuple[float, int], int]]): A list of parameter tuples for DBSCAN (eps, min_samples)
                                                           or a list of integers for KMeans (n_clusters).
        figsize (Tuple[int, int], optional): The figure size. Default is (12, 12).
        rows_cols (Tuple[int, int], optional): The number of rows and columns in the subplot grid. Default is (2, 2).
        plot_figure (bool, optional): Whether to display the plot or return the figure and axes. Default is True.

    Returns:
        Union[None, Tuple[plt.Figure, np.ndarray]]: If plot_figure is True, displays the plot and returns None.
                                                     If plot_figure is False, returns a tuple of (plt.Figure, np.ndarray of plt.Axes).

    Raises:
        ValueError: If an unsupported clustering algorithm is specified.
    """
    fig, axes = plt.subplots(rows_cols[0], rows_cols[1], figsize=figsize)
    axes = axes.flatten()

    for i, params in enumerate(param_list):
        data_copy = data.copy()
        if algorithm == "DBSCAN":
            eps, min_samples = params
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = clusterer.fit_predict(data_copy)
            n_clusters = len(np.unique(cluster_labels))
            data_copy[f"Cluster_{eps}_{min_samples}"] = cluster_labels
            data_copy[f"Cluster_{eps}_{min_samples}_plus1"] = (
                data_copy[f"Cluster_{eps}_{min_samples}"] + 1
            )
            cluster_col = f"Cluster_{eps}_{min_samples}_plus1"
            title = f"e: {eps}, s: {min_samples}, cls: {n_clusters}"
        elif algorithm == "KMeans":
            n_clusters = params
            clusterer = KMeans(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(data_copy)
            data_copy[f"Cluster_{n_clusters}"] = cluster_labels
            data_copy[f"Cluster_{n_clusters}_plus1"] = (
                data_copy[f"Cluster_{n_clusters}"] + 1
            )
            cluster_col = f"Cluster_{n_clusters}_plus1"
            title = f"{n_clusters} Clusters"
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        ax = axes[i]
        palette = sns.color_palette("hls", n_clusters)
        sns.scatterplot(
            data=data_copy,
            x="UMAP1",
            y="UMAP2",
            hue=cluster_col,
            palette=palette,
            ax=ax,
        )
        ax.set_title(title)
        ax.legend(
            loc="center right", bbox_to_anchor=(1.25, 0.5), ncol=1, title="Cluster"
        )

    if plot_figure:
        plt.tight_layout()
        plt.show()
    else:
        return fig, axes


def plot_grouped_histogram(
    data: pd.DataFrame,
    variable: str,
    group: str,
    plot_figure: bool = True,
    bins: Union[int, np.ndarray] = None,
    log_scale: bool = False,
    orientation: str = "horizontal",
    one_panel: bool = False,
) -> Union[None, Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]]:
    """
    Plot grouped histograms of a variable for different groups in a DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
        variable (str): The name of the variable column to plot the histograms for.
        group (str): The name of the group column to group the data by.
        plot_figure (bool, optional): Whether to display the plot or return the figure and axes. Default is True.
        bins (Union[int, np.ndarray], optional): The number of bins or an array of bin edges for the histograms.
                                                 If None, the bins are automatically calculated. Default is None.
        log_scale (bool, optional): Whether to use a log scale for the y-axis. Default is False.
        orientation (str, optional): The orientation of the subplots. Can be 'horizontal' or 'vertical'. Default is 'horizontal'.
        one_panel (bool, optional): Whether to plot all histograms in a single panel or separate subplots. Default is False.

    Returns:
        Union[None, Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]]:
            If plot_figure is True, displays the plot and returns None.
            If plot_figure is False and one_panel is True, returns a tuple of (plt.Figure, plt.Axes).
            If plot_figure is False and one_panel is False, returns a tuple of (plt.Figure, np.ndarray of plt.Axes).
    """
    # Get the unique groups
    groups = data[group].unique()

    # Calc. bars
    if bins is None:
        min_val = data[variable].min()
        max_val = data[variable].max()
        bin_width = (max_val - min_val) / 30  # Adjust the number of bins as needed
        bins = np.arange(min_val, max_val + bin_width, bin_width)

    # Create a single panel option
    if one_panel:
        fig, ax = plt.subplots(figsize=(6, 4))
        for g in groups:
            sns.histplot(
                data=data[data[group] == g],
                x=variable,
                ax=ax,
                label=g,
                log_scale=log_scale,
                bins=bins,
            )

        ax.set_xlabel(variable)
        ax.set_ylabel("Count")
        ax.legend(title=group)

    else:
        # Determine the number of rows and columns based on the orientation
        if orientation == "horizontal":
            nrows = 1
            ncols = len(groups)
        else:
            nrows = len(groups)
            ncols = 1

        # Create a figure with subplots for each group
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharex=True
        )

        axes = axes.ravel()

        # Plot each  histogram
        for i, g in enumerate(groups):
            if log_scale:
                sns.histplot(
                    data=data[data[group] == g],
                    x=variable,
                    ax=axes[i],
                    log_scale=True,
                    bins=bins,
                )
            else:
                sns.histplot(
                    data=data[data[group] == g], x=variable, ax=axes[i], bins=bins
                )
            axes[i].set_title(f"Group: {g}")
            axes[i].set_xlabel(variable)
            axes[i].set_ylabel("Count")

    if plot_figure:
        plt.tight_layout()
        plt.show()
    else:
        if one_panel:
            return fig, ax
        else:
            return fig, axes
