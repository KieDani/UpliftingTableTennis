'''This file is used to look at all annotatins. Sometimes the symmetry is wrong, which is why the annotations could be manually flipped to be correct.'''
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataprocessing.extract_tthq_data import video_names, additional_video_names, visible_keypoint
from paths import data_path
from uplifting.helper import table_points, world2cam, cam2img, TABLE_HEIGHT, TABLE_LENGTH, TABLE_WIDTH

# --- Configuration ---
videos_path = os.path.join(data_path, 'application_videos')
annotations_path = os.path.join(videos_path, 'table-tennis-annotations')

keypoints_to_switch = [
    [0, 5],
    [1, 4],
    [11, 12],
    [2, 3],
    [6, 7],
    [9, 10],
]

video_names = video_names + additional_video_names


# --- Helper Functions ---
def load_table_keypoints(keypoints_df):
    """
    Processes a DataFrame to extract table keypoints into a dictionary.
    Frames without full annotations are skipped.

    Args:
        keypoints_df (pd.DataFrame): DataFrame loaded from the annotation CSV.

    Returns:
        dict: A dictionary where keys are frame numbers and values are lists of
              keypoint tuples (x, y, flag).
    """
    table_detections_local = {}
    # Ensure the 'frame' column exists before proceeding
    if 'frame' not in keypoints_df.columns:
        print("Error: 'frame' column not found in CSV. Cannot process keypoints.")
        return table_detections_local

    for i, frame in enumerate(keypoints_df['frame']):
        tmp_list = []
        frame_is_annotated = True
        for keypoint_num in range(1, 14):
            # Check if all required columns exist for this keypoint
            x_col, y_col, flag_col = f'{keypoint_num:02d}_x', f'{keypoint_num:02d}_y', f'{keypoint_num:02d}_flag'
            if not all(c in keypoints_df.columns for c in [x_col, y_col, flag_col]):
                print(f"Warning: Missing columns for keypoint {keypoint_num} in frame {frame}. Skipping frame.")
                frame_is_annotated = False
                break

            x = keypoints_df[x_col][i]
            y = keypoints_df[y_col][i]
            flag = keypoints_df[flag_col][i]
            if flag == 0:  # As per user logic, flag 0 means no annotations
                frame_is_annotated = False
                break # No need to check other keypoints for this frame
            tmp_list.append((x, y, flag))

        if frame_is_annotated:  # If frame has a complete set of valid annotations
            table_detections_local[frame] = tmp_list
    return table_detections_local


def load_annotations_and_header(video_name):
    """
    Loads annotations from a CSV, skipping the header, and returns the header separately.

    Args:
        video_name (str): The name of the video

    Returns:
        tuple: A tuple containing (header_line, keypoints_dict).
               Returns (None, None) if the file can't be processed.
    """
    csv_filename = os.path.join(annotations_path, f'{video_name}_cut_keypoints.csv')
    print(f"--- Attempting to load annotations from: {csv_filename} ---")

    if not os.path.exists(csv_filename):
        print(f"Warning: Annotation file not found for {video_name}. Skipping.")
        return None, None

    try:
        # Read the first line to get the header
        with open(csv_filename, 'r') as f:
            header = f.readline()

        # Read the rest of the CSV, skipping the header
        df = pd.read_csv(csv_filename, sep=';', header=1)
        keypoints = load_table_keypoints(df)
        return header, keypoints
    except Exception as e:
        print(f"Error reading or processing {csv_filename}: {e}")
        return None, None


def save_annotations(old_df, output_path, header, final_keypoints):
    """
    Saves the final keypoints to a CSV file with the original header,
    modifying old_df in place.

    Args:
        old_df (pd.DataFrame): The original DataFrame.
        output_path (str): Path to save the new CSV file.
        header (str): The original header line.
        final_keypoints (dict): The dictionary of final annotations.
    """
    # 1. Update old_df in place with the new annotations
    for frame_number, keypoints in final_keypoints.items():
        # Get the index of the row to modify
        idx = old_df.loc[old_df['frame'] == frame_number].index[0]

        # Iterate through the keypoints and update the corresponding columns
        for i, keypoint_num in enumerate(range(1, 14)):
            x_col, y_col, flag_col = f'{keypoint_num:02d}_x', f'{keypoint_num:02d}_y', f'{keypoint_num:02d}_flag'

            # Use .loc to modify the DataFrame in-place and avoid SettingWithCopyWarning
            old_df.loc[idx, x_col] = keypoints[i][0]
            old_df.loc[idx, y_col] = keypoints[i][1]
            old_df.loc[idx, flag_col] = keypoints[i][2]

    # 2. Get the standard column names as a semicolon-separated string
    column_names = ';'.join(old_df.columns.to_list()) + '\n'
    # column_names = 'frame;ball center_x;ball center_y;ball center_flag;blur min_x;blur min_y;blur min_flag;blur max_x;blur max_y;blur max_flag;01_x;01_y;01_flag;02_x;02_y;02_flag;03_x;03_y;03_flag;04_x;04_y;04_flag;05_x;05_y;05_flag;06_x;06_y;06_flag;07_x;07_y;07_flag;08_x;08_y;08_flag;09_x;09_y;09_flag;10_x;10_y;10_flag;11_x;11_y;11_flag;12_x;12_y;12_flag;13_x;13_y;13_flag\n'

    # 3. Save the file with the custom header
    with open(output_path, 'w') as f:
        # Write the weird first line (the original header)
        f.write(header)

        # Write the standard column names
        f.write(column_names)

        # Save the updated DataFrame, but don't write the header or index
        old_df.to_csv(f, sep=';', index=False, header=False)



def change_annotations(keypoints):
    """
    Handles the logic for changing annotations for a specific frame.
    This is a placeholder for the user's implementation.

    Args:
        keypoints (list): A list of (x, y, flag) tuples for the current frame.

    Returns:
        list: The modified list of keypoint tuples.
    """
    print("\n--- Change Annotation Mode ---")
    print("Original keypoints:", keypoints)

    # --- USER IMPLEMENTATION START ---
    # This is a placeholder. For this demo, we'll change the flag
    # of the first keypoint to a new value (e.g., 99).
    modified_keypoints = keypoints.copy()
    for mapping in keypoints_to_switch:
        key1 = modified_keypoints[mapping[0]]
        key2 = modified_keypoints[mapping[1]]
        modified_keypoints[mapping[0]] = key2
        modified_keypoints[mapping[1]] = key1
    return modified_keypoints
    # --- USER IMPLEMENTATION END ---


def plotting(frame, keypoint_list, title, frame_number, video_path):
    # Draw the keypoints on the frame
    for i, (x, y, flag) in enumerate(keypoint_list):
        if flag == visible_keypoint:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"{i}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(30, 15))
    plt.imshow(frame_rgb)
    title = f"{title} Frame: {frame_number} from {os.path.basename(video_path)}"
    plt.title(title)

    # Show the plot in a non-blocking way
    plt.show(block=False)


# --- Main Application Logic ---

def main():
    """
    Main function to loop through videos and their annotated frames.
    """

    for i, video_name in enumerate(tqdm(video_names)):
        try:
            keypoints_df = pd.read_csv(os.path.join(annotations_path, f'{video_name}_cut_keypoints.csv'), sep=';', header=1)
        except FileNotFoundError:
            print(f'Annotations for video {video_name} not found. Skipping.')
            continue

        cap = cv2.VideoCapture(os.path.join(videos_path, f'{video_name}_cut.mp4'))

        video_path = os.path.join(videos_path, f'{video_name}_cut.mp4')
        header, annotations_dict = load_annotations_and_header(video_name)

        final_annotations = {}
        print(f"\n>>> Processing video: {video_path} <<<")
        print("Controls: 's' = Save/Accept, 'c' = Change, 'q' = Quit Video")

        # Loop through the annotations
        for j, frame_number in enumerate(sorted(annotations_dict.keys())):
            keypoint_list = annotations_dict[frame_number]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_number} from {video_path}. Skipping.")
                continue

            plotting(frame.copy(), keypoint_list, 'Loaded', frame_number, video_path)


            # --- Console Input ---
            # This method is more robust for remote/debugging environments.
            key = None
            while key not in ['s', 'c', 'q']:
                key = input("Enter command ('s'/'c'/'q') and press Enter: ").lower()

            # Close the figure after getting input
            plt.close()

            if key == 's':
                print(f"Frame {frame_number}: Annotation accepted.")
                final_annotations[frame_number] = keypoint_list

            elif key == 'c':
                print(f"Frame {frame_number}: Changing annotation...")
                modified_keypoints = change_annotations(keypoint_list)
                plotting(frame.copy(), modified_keypoints, 'Modified', frame_number, video_path)
                final_annotations[frame_number] = modified_keypoints

            elif key == 'q':
                print("Quitting current video.")
                break

        cap.release()
        output_path = os.path.join(annotations_path, 'corrected', f'{video_name}_cut_keypoints.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_annotations(keypoints_df, output_path, header, final_annotations)

    print(f"\n>>> Finished processing video: {video_path} <<<")



if __name__ == '__main__':
    main()