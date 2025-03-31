import plotly.graph_objects as go

def display_points_3d(dfs,
                      x_col='x',
                      y_col='y',
                      z_col='z',
                      names=None,
                      colors=None,
                      marker_size=5,
                      title="3D Plot"):
    """
    Display one or more pandas DataFrames as interactive 3D scatter plots using Plotly.

    Parameters:
      dfs (DataFrame or list of DataFrames): The DataFrame(s) to plot.
      x_col (str): Column name for x-axis data. Default is 'x'.
      y_col (str): Column name for y-axis data. Default is 'y'.
      z_col (str): Column name for z-axis data. Default is 'z'.
      names (list of str): Names for each DataFrame trace. Defaults to 'Data 1', 'Data 2', etc.
      colors (list of str): Colors for each DataFrame trace. Defaults to a preset list of colors.
      marker_size (int): Size of the markers. Default is 5.
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

    fig = go.Figure()
    for df, name, color in zip(dfs, names, colors):
        fig.add_trace(go.Scatter3d(
            x=df[x_col],
            y=df[y_col],
            z=df[z_col],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=color
            ),
            name=name
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ),
        title=title
    )

    fig.show()
