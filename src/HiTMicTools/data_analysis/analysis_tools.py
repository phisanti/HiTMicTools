import os
import pandas as pd
from typing import Any, Tuple, Union, Optional, Dict
import numpy as np


def summary_by(df: pd.DataFrame, summarise_by: str, variable: str) -> pd.DataFrame:
    """
    Summarize a variable in a DataFrame by a specified grouping variable.

    Args:
        df (pd.DataFrame): The input DataFrame.
        summarise_by (str): The column name to group the data by.
        variable (str): The column name of the variable to summarize.

    Returns:
        pd.DataFrame: A DataFrame containing summary statistics for the specified variable
        grouped by the specified grouping variable. The summary statistics include mean,
        max, min, standard deviation, 95th percentile, 5th percentile, and count.
    """
    result = df.groupby(summarise_by)[variable].agg(
        [
            ("mean", "mean"),
            ("max", "max"),
            ("min", "min"),
            ("std", "std"),
            ("quantile_95", lambda x: x.quantile(0.95)),
            ("quantile_05", lambda x: x.quantile(0.05)),
            ("n", "count"),
        ]
    )

    return result


def read_csv_files(
    folder: str, idcol: str, pattern: Optional[str] = None, **kwargs
) -> Optional[pd.DataFrame]:
    """
    Read all CSV files in a folder and combine them into a single DataFrame.

    Args:
        folder (str): Path to the folder containing the CSV files.
        idcol (str): Name of the column to add to the DataFrame, containing the file name.
        pattern (str, optional): Pattern to filter the CSV files. Only files containing the pattern will be read.
        **kwargs: Additional keyword arguments to pass to pd.read_csv().

    Returns:
        pd.DataFrame: Combined DataFrame from all the CSV files.
        None: If no CSV files are found in the specified folder.
    """
    # List csv files and filter by pattern
    csv_files = [file for file in os.listdir(folder) if file.endswith(".csv")]
    if pattern:
        csv_files = [file for file in csv_files if pattern in file]

    # Read all CSV files and combine
    df_list = []
    for file in csv_files:
        file_path = os.path.join(folder, file)
        df = pd.read_csv(file_path, **kwargs)
        df[idcol] = file
        df_list.append(df)

    # Concat
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        return combined_df
    else:
        print("No CSV files found in the specified folder.")
        return None


def cluster_binder(
    dictionary: Dict[Any, list], dataframe: pd.DataFrame, target_column: str
) -> pd.Series:
    """
    Bind cluster labels to values in a DataFrame column based on a dictionary mapping.

    Args:
        dictionary (Dict[Any, list]): Dictionary mapping cluster labels to lists of values.
        dataframe (pd.DataFrame): DataFrame containing the target column.
        target_column (str): Name of the column in the DataFrame to bind cluster labels to.

    Returns:
        pd.Series: Series containing the bound cluster labels for each value in the target column.
    """

    def find_key(value: Any) -> Optional[Any]:
        """
        Find the key in the dictionary that contains the given value.

        Args:
            value (Any): Value to search for in the dictionary.

        Returns:
            Optional[Any]: Key containing the value, or None if not found.
        """
        for key, item_list in dictionary.items():
            if value in item_list:
                return key
        return None

    new_col = dataframe[target_column].apply(find_key)

    return new_col


def create_array_from_coords(
    df: pd.DataFrame,
    img_shape: Tuple[int, int],
    value_column: str,
    coords_column: str = "coords_list",
    batch_column: Union[int, str] = 0,
    to_one_hot: bool = False,
) -> np.ndarray:
    """
    Create an array from coordinates and corresponding values.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing all the data.
        img_shape (tuple): Shape of the output array in the format (height, width).
        value_column (str): Name of the column in df containing the values for each coordinate.
        coords_column (str): Name of the column in df containing the coordinate tuples (x, y). Default is 'coords_list'.
        batch_column (Union[int, str]): If int, all coordinates are assigned to this batch. If str, name of the column in df containing the batch indices for each coordinate. Default is 0.
        to_one_hot (bool): Whether to convert the array to one-hot encoding. Default is False.

    Returns:
        np.ndarray: Array with values assigned to the specified coordinates.
    """

    # Prepare data
    value_column = df[value_column]
    coords_column = df[coords_column]

    if batch_column == 0:
        batch_column = [0] * len(coords_column)
    else:
        batch_column = df[batch_column]

    num_batches = max(batch_column) + 1
    array_stack = np.zeros((num_batches,) + img_shape)
    num_classes = len(np.unique(value_column))

    # Iterate over coords
    for batch, coords, value in zip(batch_column, coords_column, value_column):
        x_coords, y_coords = zip(*coords)
        array_stack[batch, x_coords, y_coords] = value

    # Return one-hot-encoded data if requested
    if to_one_hot:
        value_column = value_column.astype(int)
        array_stack = one_hot_encode(array_stack, num_classes)

    # Adjust dtype for memory efficiency
    if num_classes < 256 or to_one_hot:
        array_stack = array_stack.astype(np.uint8)
    elif num_classes < 65536:
        array_stack = array_stack.astype(np.uint16)
    else:
        array_stack = array_stack.astype(np.uint32)

    return array_stack


def one_hot_encode(image_stack, num_classes):
    """
    Transform multiclass image stack to one-hot encoding. Classes must be integers and channel dimension
    will be used to encode the different classes.

    Args:
        image_stack (numpy.ndarray): Image stack with integer values representing classes.
        num_classes (int): Number of classes in the image stack.

    Returns:
        numpy.ndarray: One-hot encoded image stack with the channel dimension moved to the second position.
    """
    identity = np.eye(num_classes)
    one_hot_encoded = identity[image_stack]
    one_hot_encoded = np.moveaxis(one_hot_encoded, -1, 1)

    return one_hot_encoded


def create_array_from_coords_old(
    df: pd.DataFrame,
    x_dim: int,
    y_dim: int,
    z_dim: int,
    z_col: str,
    coord_col: str = "coords_list",
) -> np.ndarray:
    """
    Create a 3D numpy array from coordinates stored in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the coordinates and corresponding frame (z) values.
        x_dim (int): Dimension of the array along the x-axis.
        y_dim (int): Dimension of the array along the y-axis.
        z_dim (int): Dimension of the array along the z-axis.
        z_col (str): Name of the column in the DataFrame containing the frame (z) values.
        coord_col (str, optional): Name of the column in the DataFrame containing the coordinate tuples (x, y). Default is 'coords_list'.

    Returns:
        np.ndarray: 3D numpy array with values set to 1 at the specified coordinates.
    """
    # Create a black numpy array of dimensions x, y, z
    array = np.zeros((z_dim, x_dim, y_dim), dtype=float)

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # Get the x, y coordinates and the corresponding frame (z)
        coords = row[coord_col]
        frame = row[z_col]
        # Update the pixel values in the array
        for coord in coords:
            x, y = coord
            array[frame, x, y] = 1

    return array
