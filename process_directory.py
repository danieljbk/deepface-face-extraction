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
    crop_and_save_detected_faces,
    find_most_similar_face,  # Assuming this function exists
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
    """Select the most similar face per directory."""
    similar_faces["directory"] = similar_faces["identity"].apply(
        lambda x: os.path.dirname(x)
    )
    return (
        similar_faces.sort_values("distance").groupby("directory").first().reset_index()
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

        return similar_faces["identity"].tolist()
    except Exception as e:
        logging.error(f"Failed to find similar faces: {e}")
        return []


def process_directory(
    directory,
    reference_img_path,
    similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
    unique_per_image=DEFAULT_UNIQUE_PER_IMAGE,
):
    all_detected_faces_dir = os.path.join(directory, "all_detected_faces")
    ensure_directory_exists(all_detected_faces_dir)

    # Face detection and cropping logic here...
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(root, file)
                logging.info(f"Processing image: {img_path}")

                # Detect faces in the image
                faces = detect_faces(img_path)
                if not faces:
                    logging.warning(f"No faces detected in {img_path}. Skipping.")
                    continue

                # Crop and save detected faces
                crop_and_save_detected_faces(faces, img_path)

    # currently, crop_and_save_detected_faces() saves to cropped-face-db/person/img_path
    # in contrast to this function, in which we make a new dir for output in the original dir
    # so we must run DeepFace.find on that directory
    detected_faces_dir = os.path.join(CROPPED_IMG_DB_NAME, BASE_IMG_DIR_NAME)

    # Find similar faces with the specified options
    similar_faces = find_similar_faces(
        detected_faces_dir,
        reference_img_path,
        similarity_threshold,
        unique_per_image,
    )

    # so far so good
    if similar_faces:
        logging.info(f"Found similar faces: {similar_faces}")

        # need new names
        for img_path in similar_faces:
            # Extract the folder name (which contains the original image name) from the img_path
            original_img_name = os.path.basename(os.path.dirname(img_path))

            # Set the destination path for copying the photo using all_detected_faces_dir
            new_most_similar_cropped_img_path = os.path.join(
                all_detected_faces_dir,
                f"{original_img_name}-{os.path.basename(img_path)}",  # Use folder name in the new file name
            )

            # Copy & save the most similar cropped image
            most_similar_cropped_img = load_and_process_image(
                img_path
            )  # Corrected variable name
            save_processed_image(
                new_most_similar_cropped_img_path, most_similar_cropped_img
            )
            logging.info(
                f"Copied and saved the most similar face to: ({new_most_similar_cropped_img_path})"
            )
    else:
        logging.info(f"No similar faces found in directory: {directory}.")


if __name__ == "__main__":
    configure_logging()
    BASE_IMG_DIR_PATH = os.path.join(BASE_IMG_DB_NAME, BASE_IMG_DIR_NAME)
    process_directory(BASE_IMG_DIR_PATH, REFERENCE_IMG_FILE_PATH)
