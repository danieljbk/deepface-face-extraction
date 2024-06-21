import os
import cv2
import logging
from deepface import DeepFace
from utils import load_and_process_image, save_processed_image, ensure_directory_exists
from config import (
    BASE_IMG_FILE_PATH,
    BASE_IMG_DIR_NAME,
    CROPPED_IMG_DB_NAME,
    REFERENCE_IMG_FILE_PATH,
    configure_logging,
)
import pandas as pd


def crop_and_save_detected_faces(faces: list, img_path: str):
    detected_faces_dir = os.path.join(
        CROPPED_IMG_DB_NAME,
        BASE_IMG_DIR_NAME,
        os.path.basename(img_path) + "/",
    )  # the "/" is added to ensure this is recognized as a directory path and not a file
    ensure_directory_exists(detected_faces_dir)

    # load image here so I simply crop this loaded image for every detected face
    img = load_and_process_image(img_path)
    if img is None:
        logging.error(f"Failed to load image at ({img_path}), aborting face detection.")
        return None  # Exit the function if image is not loaded

    image_height = img.shape[0]
    image_width = img.shape[1]

    # used to create unique filename for each face (1.png, 2.png, etc.)
    count = 1
    for face in faces:
        # Extract facial area data from deepface results
        facial_area = face["facial_area"]
        x, y, w, h = (
            facial_area["x"],
            facial_area["y"],
            facial_area["w"],
            facial_area["h"],
        )

        # Correcting for out-of-bound coordinates
        # Because DeepFace adds padding to images, the x or y value can be negative. Let's fix this.
        # I'll assume the same can happen for the opposite end, where the value exceeds the image area.
        x = max(0, x)
        y = max(0, y)
        w = min(w, image_width - x)
        h = min(h, image_height - y)

        # Crop the image using the facial area coordinates
        cropped_img = img[y : y + h, x : x + w]
        if cropped_img.size == 0:
            logging.error("Error: Cropped image is empty.")
        else:
            # Set up the directory and filename for the cropped image
            cropped_img_filename = f"{count}.png"
            cropped_img_full_path = os.path.join(
                detected_faces_dir, cropped_img_filename
            )
            success = save_processed_image(cropped_img_full_path, cropped_img)
            if success:
                count += 1

    # there was at least one face image saved
    if count > 1:
        return detected_faces_dir
    else:
        logging.error(
            f"Unexpected Error: No faces saved to ({detected_faces_dir}), aborting..."
        )
        return None


# For process_directory.py, a different function, find_similar_faces(), is used.
def find_most_similar_face(
    directory,
):
    """Handles multiple face detections by selecting the most similar face (lowest distance value)"""
    try:
        # NOTE: DeepFace.find generates a .pkl file in the directory, allowing potential usage in subsequent runs. See CLI output example:
        # "There are now 1 representations in ds_model_vggface_detector_retinaface_aligned_normalization_base_expand_0.pkl"
        dfs = DeepFace.find(
            img_path=REFERENCE_IMG_FILE_PATH,
            db_path=directory,
            detector_backend="retinaface",
        )  # these are dataframes, ordered by most similar to least similar

        # TODO: Investigate why DeepFace.find returns this single dataframe in a list.
        dfs = dfs[0]
    except ValueError as e:
        logging.error("Failed to find any similar faces, aborting...")

    # The following assumes the individual's face appears no more than once in photo (no collages allowed)
    try:
        # dfs could be an empty dataframe here if deepface determined that NONE of the faces were similar to the reference.
        # This is why the reference image is quite important. This can lead to complexity...
        # if there are photos of person A from a long time ago, that are compared with recent photos.
        # but, this could also be used as a feature if we can filter photos by distance and automatically group them...
        # into a batch of photos for the face recognized as "old photos of A" v.s. "recent photos of A"
        most_similar_df = dfs.loc[0]
        logging.info(
            f"Found at least 1 face similar to reference image ({REFERENCE_IMG_FILE_PATH}) in ({directory})."
        )

        # extract path data from dataframes
        most_similar_face_img_path = most_similar_df["identity"]

        return most_similar_face_img_path
    except IndexError as e:
        logging.error(
            "No similar faces found, or an error occurred in the DataFrame indexing."
        )
        return None
    except Exception as e:
        logging.error(f"Unexpected error when finding most similar face: {e}")
        return None


def detect_faces(img_path):
    try:
        # The most capable "backend" provided by DeepFace for face detection is "retinaface". The creator of deepface alternatively recommends "mtcnn".
        faces = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend="retinaface",
        )
    except ValueError as e:
        logging.error(f"Face could not be detected in ({img_path}).")
        return None

    return faces


def process_single_image(img_path):
    logging.info(f"Starting processing of image: {img_path}")
    faces = detect_faces(img_path)
    if not faces:
        logging.error(f"No faces detected in {img_path}. Skipping.")
        return

    detected_faces_dir = crop_and_save_detected_faces(faces, img_path)
    if not detected_faces_dir:
        logging.error(f"No faces saved from {img_path}. Skipping.")
        return

    most_similar_face_img_path = find_most_similar_face(detected_faces_dir)
    if most_similar_face_img_path:
        logging.info(f"Most similar face found at: {most_similar_face_img_path}")

        # Set the destination path for copying the photo
        new_most_similar_cropped_img_path = os.path.join(
            CROPPED_IMG_DB_NAME,
            BASE_IMG_DIR_NAME,
            "cropped_" + os.path.basename(img_path),
        )

        # Copy & save the most similar cropped image
        most_similar_cropped_img = load_and_process_image(most_similar_face_img_path)
        save_processed_image(
            new_most_similar_cropped_img_path, most_similar_cropped_img
        )
        logging.info(
            f"Copied and saved the most similar face to: ({new_most_similar_cropped_img_path})"
        )
    else:
        logging.info(f"No similar faces found for image: {img_path}")

    logging.info(f"Completed processing of image: {img_path}")


# For standalone testing of this file.
if __name__ == "__main__":
    configure_logging()
    test_img_path = BASE_IMG_FILE_PATH
    process_single_image(test_img_path)
