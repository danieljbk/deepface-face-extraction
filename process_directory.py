import os
import logging
import pandas as pd
from termcolor import colored
from deepface import DeepFace
from utils import load_and_process_image, save_processed_image, ensure_directory_exists
from config import (
    BASE_IMG_DB_NAME,
    BASE_IMG_DIR_NAME,
    CROPPED_IMG_DB_NAME,
    REFERENCE_IMG_FILE_PATH,
    configure_logging,
    FAILED_TO_FIND_FACES_LOG,
    FAILED_TO_LOAD_IMAGE_LOG,
    CROPPED_IMAGE_EMPTY_LOG,
    NO_FACES_SAVED_LOG,
    FOUND_SIMILAR_FACES_LOG,
    CROPPED_SAVED_FACE_LOG,
    NO_SIMILAR_FACES_FOUND_LOG,
)

# Define default values as constants at the module level
DEFAULT_SIMILARITY_THRESHOLD = 0.6
DEFAULT_UNIQUE_PER_IMAGE = True


def visualize_dataframes(df, header="DataFrame Visualization"):
    """Prints the DataFrame column names in groups, followed by values for each row in a visually distinct format with color coding."""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", None)

    print(colored(f"\n--- {header} ---", "cyan", attrs=["bold"]))

    # Define column groups
    groups = {
        "Identity": ["identity"],
        "Hash": ["hash"],
        "Target Coordinates": ["target_x", "target_y", "target_w", "target_h"],
        "Source Coordinates": ["source_x", "source_y", "source_w", "source_h"],
        "Metrics": ["threshold", "distance"],
    }

    # Print column headers grouped
    for group_name, columns in groups.items():
        grouped_columns = ", ".join(
            [colored(col, "yellow") for col in columns if col in df.columns]
        )
        if grouped_columns:
            print(f"{group_name}: {grouped_columns}")

    # Print each row's values grouped
    for index, row in df.iterrows():
        print(colored(f"Row {index}:", "cyan", attrs=["bold"]))
        for group_name, columns in groups.items():
            values = ", ".join(
                [colored(str(row[col]), "green") for col in columns if col in row]
            )
            if values:
                print(f"  {group_name}: {values}")

    print(colored("--- End of Visualization ---", "cyan", attrs=["bold"]))


def filter_faces_by_similarity(dfs, similarity_threshold):
    """Filter faces based on the similarity threshold."""
    return dfs[dfs["distance"] < similarity_threshold]


def select_unique_faces(similar_faces):
    """Select the most similar face per image file."""
    # Group by the image file path directly to distinguish faces from each image
    return (
        similar_faces.sort_values("distance")
        .groupby("identity")
        .first()
        .reset_index()  # The reset_index() method is crucial when you need to ensure that the DataFrame has a standard integer index, especially after operations like groupby().first() which can leave the DataFrame with a non-standard index.
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
        if not isinstance(dfs, pd.DataFrame):
            dfs = pd.concat(dfs)  # Only concatenate if dfs is not already a DataFrame

        visualize_dataframes(dfs)

        similar_faces = filter_faces_by_similarity(dfs, similarity_threshold)
        visualize_dataframes(similar_faces)

        if unique_per_image:
            similar_faces = select_unique_faces(similar_faces)

        visualize_dataframes(similar_faces)

        return similar_faces
    except Exception as e:
        logging.error(
            FAILED_TO_FIND_FACES_LOG.format(img_path=reference_img_path, error=str(e))
        )
        return []


def crop_face(img, x, y, w, h):
    """Crop the face from the image based on provided coordinates."""
    image_height, image_width = img.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, image_width - x)
    h = min(h, image_height - y)
    return img[y : y + h, x : x + w]


def save_face_image(directory, img, count):
    """Save the cropped face image to the specified directory."""
    cropped_img_filename = f"{count}.png"
    cropped_img_full_path = os.path.join(directory, cropped_img_filename)
    return save_processed_image(cropped_img_full_path, img)


def crop_and_save_detected_faces(faces: list, img_path: str):
    detected_faces_dir = os.path.join(
        CROPPED_IMG_DB_NAME, BASE_IMG_DIR_NAME, os.path.basename(img_path) + "/"
    )
    ensure_directory_exists(detected_faces_dir)

    img = load_and_process_image(img_path)
    if img is None:
        logging.error(FAILED_TO_LOAD_IMAGE_LOG.format(img_path=img_path))
        return None

    count = 1
    for x, y, w, h in faces:
        cropped_img = crop_face(img, x, y, w, h)
        if cropped_img.size == 0:
            logging.error(CROPPED_IMAGE_EMPTY_LOG)
            continue

        if save_face_image(detected_faces_dir, cropped_img, count):
            count += 1

    if count > 1:
        return detected_faces_dir
    else:
        logging.error(NO_FACES_SAVED_LOG.format(detected_faces_dir=detected_faces_dir))
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
    if similar_faces.empty:
        logging.info(NO_SIMILAR_FACES_FOUND_LOG.format(directory=directory))
        return

    logging.info(FOUND_SIMILAR_FACES_LOG.format(directory=directory))

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

        logging.info(CROPPED_SAVED_FACE_LOG.format(img_path=img_path))


if __name__ == "__main__":
    configure_logging()
    BASE_IMG_DIR_PATH = os.path.join(BASE_IMG_DB_NAME, BASE_IMG_DIR_NAME)
    process_directory(BASE_IMG_DIR_PATH, REFERENCE_IMG_FILE_PATH)
