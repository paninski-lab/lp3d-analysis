from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def plot_labeled_frames(
    csv_files: dict,
    data_dir: str,
    s: float = 1.0,
    linewidth: float = 1.0,
    txt_offset: int = 0,
    height: float = 6.0,
    skeleton=None,
    skeleton_colors=None
) -> None:
    """Plot labels from a csv file on top of frames for a multiview dataset.

    Parameters
    ----------
    csv_files: list of strings, each entry corresponds to labels from a single view
    data_dir: base data dir where csv_files and frames are saved
    s: marker size
    linewidth: skeleton line width
    txt_offset: offset for textboxes
    height: height of figure
    skeleton: list of lists, each element is a list [idx0, idx1] which defines two nodes to connect
    skeleton_colors: str or list of strs; if str, single color for skeleton; if list of strs, one
        for each element of `skeleton`

    """
    data_dir = Path(data_dir)

    cam_names = list(csv_files.keys())
    base_cam_name = cam_names[0]

    # Loop through the rows
    for index in csv_files[base_cam_name].index:  # All files have the same indices
        fig, axes = plt.subplots(2, 3, figsize=(12, height))
        axes = axes.flatten()

        # Loop through cameras
        for i, cam_name in enumerate(cam_names):

            index_ = index.replace(base_cam_name, cam_name)
            # Get the corresponding row
            row = csv_files[cam_name].loc[index_]

            # Extract x, y coordinates (alternating columns)
            x_coords = row.iloc[0::2]
            y_coords = row.iloc[1::2]

            # Get the original column names
            columns = csv_files[cam_name].columns[0::2]  # Get x-coordinates' column names

            # Load the corresponding image
            image_path = data_dir / index_  # index is the relative path to the image
            image = cv2.imread(str(image_path))

            # Check if the image loaded successfully
            if image is None:
                print(f"Error: Could not load image {image_path}")
                continue

            # Convert BGR to RGB for plotting
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Plot the image
            axes[i].imshow(image)
            axes[i].axis('off')  # Turn off axes

            # plot labels
            if skeleton_colors:
                color = 'w'
            else:
                color = 'r'
            axes[i].scatter(x_coords, y_coords, color=color, s=s)

            # plot skeleton
            for l, limb in enumerate(skeleton):
                if skeleton_colors:
                    color = skeleton_colors[l]
                else:
                    color = 'r'
                axes[i].plot(x_coords.iloc[limb], y_coords.iloc[limb], color, linewidth=linewidth)

            # Add the camera name in the top-left corner
            axes[i].text(
                txt_offset, txt_offset, cam_name,  # Position: pixels from the top-left
                color='white', fontsize=12, weight='bold',
                ha='left', va='top',
                bbox=dict(facecolor='black', alpha=0.8)  # , boxstyle='round,pad=0.3')
            )

        # Remove all possible whitespace
        fig.subplots_adjust(left=0, right=1, top=0.98, bottom=0, wspace=0, hspace=0)

        # Add a suptitle with the row index
        fig.suptitle(index.replace(f'_{base_cam_name}', ''), fontsize=16)

        plt.tight_layout()

        # save labeled frame in another directory
        pieces = index.split('/')
        pieces[0] = pieces[0] + '-check'
        pieces[1] = pieces[1].replace(f'_{base_cam_name}', '')
        title = '/'.join(pieces)

        save_file = Path(data_dir) / title
        save_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_file, bbox_inches='tight', dpi=300)
        plt.close(fig)
