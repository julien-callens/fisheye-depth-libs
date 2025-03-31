import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def process_camera_images(files,
                          lower_color=np.array([0, 100, 0], dtype=np.uint8),
                          upper_color=np.array([255, 255, 255], dtype=np.uint8)):
    """
    Process a list of image file paths and extract the median (u, v)
    coordinates of pixels that fall within the specified green range.

    Displays a progress bar while processing.

    Parameters:
      files (list of str): List containing full paths to camera images.
      lower_color (np.array): Lower bounds of the green color range.
      upper_color (np.array): Upper bounds of the green color range.

    Returns:
      pd.DataFrame: A DataFrame with the columns: image_index, file_name, u, v.
    """
    results = []

    for i, file in enumerate(tqdm(files, desc="Processing images")):
        image = cv2.imread(file)

        if image is None:
            print(f"Warning: Unable to read image at {file}. Skipping.")
            continue

        green_mask = (
                (image[:, :, 0] >= lower_color[0]) & (image[:, :, 0] <= upper_color[0]) &
                (image[:, :, 1] >= lower_color[1]) & (image[:, :, 1] <= upper_color[1]) &
                (image[:, :, 2] >= lower_color[2]) & (image[:, :, 2] <= upper_color[2])
        )

        green_coords = np.column_stack(np.where(green_mask))

        if green_coords.size > 0:
            green_coords[:, 0] = image.shape[0] - green_coords[:, 0]
            median_u = np.median(green_coords[:, 1])
            median_v = np.median(green_coords[:, 0])
        else:
            median_u = np.nan
            median_v = np.nan

        results.append({
            'frame_index': i,
            'file_name': os.path.basename(file),
            'u': median_u,
            'v': median_v
        })

    return pd.DataFrame(results)

def process_video_frames(video_path,
                         lower_color=np.array([0, 100, 0], dtype=np.uint8),
                         upper_color=np.array([255, 255, 255], dtype=np.uint8)):
    """
    Process all frames in the specified video and extract the median (u, v) coordinates
    of pixels that fall within the specified color range.

    Parameters:
      - video_path (str): Path to the video file.
      - lower_color (np.ndarray): Lower bound of the color range (in BGR).
      - upper_color (np.ndarray): Upper bound of the color range (in BGR).

    Returns:
      - pd.DataFrame: A DataFrame with the columns: frame_index, u, v.

    Practical example:
      To extract red ball coordinates (using BGR color bounds for red):

          lower_red = np.array([0, 0, 100], dtype=np.uint8)
          upper_red = np.array([75, 75, 255], dtype=np.uint8)
          df = process_video_frames('path/to/video.mp4', lower_red, upper_red)

    Note:
      - The function uses a progress bar (via tqdm) if the total frame count is available.
      - It flips the y-axis to match an origin at the bottom-left, similar to the image parser.
    """
    import os
    from tqdm import tqdm
    import cv2
    import numpy as np
    import pandas as pd

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = []
    frame_index = 0

    with tqdm(total=total_frames, desc=f"Processing frames in {os.path.basename(video_path)}") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            mask = (
                    (frame[:, :, 0] >= lower_color[0]) & (frame[:, :, 0] <= upper_color[0]) &
                    (frame[:, :, 1] >= lower_color[1]) & (frame[:, :, 1] <= upper_color[1]) &
                    (frame[:, :, 2] >= lower_color[2]) & (frame[:, :, 2] <= upper_color[2])
            )
            color_coords = np.column_stack(np.where(mask))

            if color_coords.size > 0:
                color_coords[:, 0] = frame.shape[0] - color_coords[:, 0]
                median_u = np.median(color_coords[:, 1])
                median_v = np.median(color_coords[:, 0])
            else:
                median_u = np.nan
                median_v = np.nan

            results.append({
                'frame_index': frame_index,
                'u': median_u,
                'v': median_v
            })
            frame_index += 1
            pbar.update(1)

    cap.release()
    return pd.DataFrame(results)
