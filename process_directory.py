import os
import logging
import pandas as pd
from deepface import DeepFace
from config import (
    BASE_IMG_DB_NAME,
    BASE_IMG_DIR_NAME,
    CROPPED_IMG_DB_NAME,
    REFERENCE_IMG_FILE_PATH,
    configure_logging,
)
from process_image import (
    detect_faces,
)

from utils import load_and_process_image, save_processed_image, ensure_directory_exists

# Define default values as constants at the module level
DEFAULT_SIMILARITY_THRESHOLD = 0.6
DEFAULT_UNIQUE_PER_IMAGE = True


# currently just prints the dataframe to the terminal
def visualize_dataframes(df):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", None)
    print(df)


def filter_faces_by_similarity(dfs, similarity_threshold):
    """Filter faces based on the similarity threshold."""
    return dfs[dfs["distance"] < similarity_threshold]


def select_unique_faces(similar_faces):
    """Select the most similar face per image file."""
    # Group by the image file path directly to distinguish faces from each image
    return (
        similar_faces.sort_values("distance").groupby("identity").first().reset_index()
    )


def find_similar_faces(
    directory,
    reference_img_path,
    similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
    unique_per_image=DEFAULT_UNIQUE_PER_IMAGE,
):
    try:
        dfs = DeepFace.find(
            img_path=reference_img_path,
            db_path=directory,
            detector_backend="retinaface",
        )
        if isinstance(dfs, list):
            dfs = pd.concat(dfs)  # Assuming dfs is a list of DataFrames

        visualize_dataframes(dfs)

        similar_faces = filter_faces_by_similarity(dfs, similarity_threshold)
        visualize_dataframes(similar_faces)

        if unique_per_image:
            similar_faces = select_unique_faces(similar_faces)

        visualize_dataframes(similar_faces)

        return similar_faces
    except Exception as e:
        logging.error(f"Failed to find similar faces: {e}")
        return []


def crop_and_save_detected_faces(faces: list, img_path: str):
    detected_faces_dir = os.path.join(
        CROPPED_IMG_DB_NAME, BASE_IMG_DIR_NAME, os.path.basename(img_path) + "/"
    )
    ensure_directory_exists(detected_faces_dir)

    # Load the image once to crop all detected faces
    img = load_and_process_image(img_path)
    if img is None:
        logging.error(f"Failed to load image at ({img_path}), aborting face detection.")
        return None

    image_height, image_width = img.shape[:2]

    # Used to create unique filename for each face (1.png, 2.png, etc.)
    count = 1
    for x, y, w, h in faces:
        # Correcting for out-of-bound coordinates
        x = max(0, x)
        y = max(0, y)
        w = min(w, image_width - x)
        h = min(h, image_height - y)

        # Crop the image using the facial area coordinates
        cropped_img = img[y : y + h, x : x + w]
        if cropped_img.size == 0:
            logging.error("Error: Cropped image is empty.")
            continue

        # Set up the directory and filename for the cropped image
        cropped_img_filename = f"{count}.png"
        cropped_img_full_path = os.path.join(detected_faces_dir, cropped_img_filename)
        success = save_processed_image(cropped_img_full_path, cropped_img)
        if success:
            count += 1

    # Check if at least one face image was saved
    if count > 1:
        return detected_faces_dir
    else:
        logging.error(
            f"Unexpected Error: No faces saved to ({detected_faces_dir}), aborting..."
        )
        return None


def process_directory(
    directory,
    reference_img_path,
    similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
    unique_per_image=DEFAULT_UNIQUE_PER_IMAGE,
):
    ensure_directory_exists(directory)

    # Find similar faces with the specified options
    similar_faces = find_similar_faces(
        directory,
        reference_img_path,
        similarity_threshold,
        unique_per_image,
    )

    # similar_faces is a dataframe with columns "identity", "distance", "threshold"
    # and face coordinates ("target_x", "target_y", "target_w", "target_h")
    if not similar_faces.empty:
        logging.info(f"Found similar faces: {similar_faces['identity'].tolist()}")

        for _, row in similar_faces.iterrows():
            img_path = row["identity"]
            # Coordinates for cropping the detected face
            x, y, w, h = (
                row["target_x"],
                row["target_y"],
                row["target_w"],
                row["target_h"],
            )
            # Crop and save detected faces using the coordinates provided by DeepFace.find
            crop_and_save_detected_faces([(x, y, w, h)], img_path)

            logging.info(f"Cropped and saved face from: {img_path}")
    else:
        logging.info(f"No similar faces found in directory: {directory}.")


if __name__ == "__main__":
    configure_logging()
    BASE_IMG_DIR_PATH = os.path.join(BASE_IMG_DB_NAME, BASE_IMG_DIR_NAME)
    process_directory(BASE_IMG_DIR_PATH, REFERENCE_IMG_FILE_PATH)
