import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for enabling 3D plotting

def display_points_3d_matplotlib(dfs,
                                 x_col='x',
                                 y_col='y',
                                 z_col='z',
                                 names=None,
                                 colors=None,
                                 marker='o',
                                 marker_size=50,
                                 figure_size=(10, 8),
                                 title="3D Plot"):
    """
    Display one or more pandas DataFrames as interactive 3D scatter plots using Matplotlib.

    Parameters:
      dfs (DataFrame or list of DataFrames): The DataFrame(s) to plot.
      x_col (str): Column name for x-axis data. Default is 'x'.
      y_col (str): Column name for y-axis data. Default is 'y'.
      z_col (str): Column name for z-axis data. Default is 'z'.
      names (list of str): Labels for each DataFrame trace. Defaults to 'Data 1', 'Data 2', etc.
      colors (list of str): Colors for each DataFrame trace. Defaults to a preset list of colors.
      marker (str): Marker style for the scatter points. Default is 'o'.
      marker_size (int): Size of the markers (area in points^2). Default is 50.
      figure_size (tuple): Size of the figure (width, height). Default is (10, 8).
      title (str): Title of the plot.

    Returns:
      Displays an interactive 3D scatter plot.
    """
    if not isinstance(dfs, list):
        dfs = [dfs]

    if names is None:
        names = [f"Data {i+1}" for i in range(len(dfs))]

    if colors is None:
        default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        colors = [default_colors[i % len(default_colors)] for i in range(len(dfs))]

    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, projection='3d')

    for df, name, color in zip(dfs, names, colors):
        ax.scatter(df[x_col], df[y_col], df[z_col], c=color, marker=marker, s=marker_size, label=name)

    ax.set_xlabel(f'{x_col} coordinate')
    ax.set_ylabel(f'{y_col} coordinate')
    ax.set_zlabel(f'{z_col} coordinate')
    ax.set_title(title)

    ax.legend()

    plt.show()
