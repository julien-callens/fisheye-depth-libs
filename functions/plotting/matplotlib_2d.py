from matplotlib import pyplot as plt


def visualize_points(df1, df2,
                     label1='camL',
                     label2='camR',
                     resolution=(720, 720)):
    """
    Visualize median (u, v) coordinates for two sets of camera data.

    Parameters:
      :param df1: DataFrame containing columns 'u' and 'v' for first camera.
      :param df2: DataFrame containing columns 'u' and 'v' for second camera.
      :param label1: Label for the first camera points.
      :param label2: Label for the second camera points.
      :param resolution:
    """

    plt.figure(figsize=(resolution[0] / 100, resolution[1] / 100), dpi=100)

    plt.scatter(df1['u'], df1['v'], color='blue', marker='o', label=label1)
    plt.scatter(df2['u'], df2['v'], color='red', marker='x', label=label2)

    plt.xlabel('u (Horizontal coordinate)')
    plt.ylabel('v (Vertical coordinate)')
    plt.title('Coordinates on sensors')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.xlim(0, resolution[0])
    plt.ylim(0, resolution[1])

    plt.show()
