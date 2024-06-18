import os
import cv2
import logging
from deepface import DeepFace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
)

# Constants
IMG_DB_NAME = "face-db"
IMG_DIR_NAME = "chaewon-test"
IMG_FILE_NAME = "2.jpg"
REFERENCE_IMG_FILE_NAME = "3.jpg"  # this image is what the faces are compared to & therefore must be high quality.

# Combine the directory with the base path
FULL_IMG_PATH = os.path.join(IMG_DB_NAME, IMG_DIR_NAME, IMG_FILE_NAME)


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        logging.error(f"Error: Failed to load image from {path}.")
        return None  # Explicitly return None to signal failure
    return img


def save_image(path: str, img):
    success = cv2.imwrite(path, img)
    if success:
        logging.info(f"Saved photo to {path}")
    else:
        logging.error(f"Failed to save photo to {path}")


def save_detected_faces(faces: list):
    cropped_img_db_name = "cropped-face-db"
    current_cropped_faces_dir_in_cropped_img_db_path = os.path.join(
        cropped_img_db_name, IMG_DIR_NAME, IMG_FILE_NAME + "/"
    )  # the "/" is added to ensure this is recognized as a directory path and not a file

    # Ensure the output directory exists before saving
    if not os.path.exists(current_cropped_faces_dir_in_cropped_img_db_path):
        os.makedirs(current_cropped_faces_dir_in_cropped_img_db_path)

    # load image here so I simply crop this image for every detected face
    img = load_image(FULL_IMG_PATH)
    if img is None:
        logging.error("Failed to load image, aborting face detection.")
        return None  # Exit the function if image is not loaded

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

        # Crop the image using the facial area coordinates
        cropped_img = img[y : y + h, x : x + w]
        if cropped_img.size == 0:
            logging.error("Error: Cropped image is empty.")
        else:
            # Set up the directory and filename for the cropped image
            cropped_img_filename = f"{count}.png"
            cropped_img_full_path = os.path.join(
                current_cropped_faces_dir_in_cropped_img_db_path, cropped_img_filename
            )

            save_image(cropped_img_full_path, cropped_img)
            count += 1

    return cropped_img_db_name, current_cropped_faces_dir_in_cropped_img_db_path


# Handles multiple face detections by selecting the most similar face (lowest distance value)
def find_most_similar_face(
    REFERENCE_IMG_FILE_NAME,
    IMG_DB_NAME,
    IMG_DIR_NAME,
    current_cropped_faces_dir_in_cropped_img_db_path,
):
    reference_img_path = os.path.join(
        IMG_DB_NAME, IMG_DIR_NAME, REFERENCE_IMG_FILE_NAME
    )

    # NOTE: DeepFace.find generates a .pkl file in the directory, allowing potential usage in subsequent runs. See CLI output example:
    # "There are now 1 representations in ds_model_vggface_detector_retinaface_aligned_normalization_base_expand_0.pkl"
    dfs = DeepFace.find(
        img_path=reference_img_path,
        db_path=current_cropped_faces_dir_in_cropped_img_db_path,
        detector_backend="retinaface",
    )  # these are dataframes, ordered by most similar to least similar

    # TODO: Investigate why DeepFace.find returns this single dataframe in a list.
    dfs = dfs[0]

    # useful for most cases, where the individual will only appear once in the photo
    try:
        # dfs could be an empty dataframe here...
        # if deepface determined that NONE of the faces were similar to the reference.
        # this is why the reference is quite important.
        # and this can lead to complexity...
        # if there are photos of person A from a long time ago, that are compared with recent photos.
        # but, this could also be used as a feature...
        # if we can filter photos by distance and automatically group them...
        # into a batch of photos for the face recognized as "old photos of A" v.s. "recent photos of A"
        most_similar_df = dfs.loc[0]

        logging.info("Found at least 1 face similar to reference face.")

        # extract path data from dataframes
        most_similar_cropped_img_path = most_similar_df["identity"]

        return most_similar_cropped_img_path
    except IndexError as e:
        logging.error(
            "No similar faces found, or an error occurred in the DataFrame indexing."
        )
        return None
    except Exception as e:
        logging.error(f"Unexpected error when finding most similar face: {e}")
        return None


def detect_multiple_faces(
    IMG_DB_NAME: str, IMG_DIR_NAME: str, IMG_FILE_NAME: str, FULL_IMG_PATH: str
):
    """
    The following list contains all available "backends" provided by deepface.
    "retinaface" is the most capable one for face detection.
    the creator of deepface also recommended "mtcnn", but "retinaface" is great so we will stick to it.
    """
    backends = [
        "opencv",
        "ssd",
        "dlib",
        "mtcnn",
        "fastmtcnn",
        "retinaface",
        "mediapipe",
        "yolov8",
        "yunet",
        "centerface",
    ]

    faces = DeepFace.extract_faces(
        img_path=FULL_IMG_PATH,
        detector_backend="retinaface",
    )
    return faces


faces = detect_multiple_faces(IMG_DB_NAME, IMG_DIR_NAME, IMG_FILE_NAME, FULL_IMG_PATH)
detected_faces_info = save_detected_faces(faces)

# Check if the function returned None (indicating an error)
if detected_faces_info is None:
    logging.error("Failed to detect and save faces, aborting further processing.")
else:
    cropped_img_db_name, current_cropped_faces_dir_in_cropped_img_db_path = (
        detected_faces_info
    )

    most_similar_cropped_img_path = find_most_similar_face(
        REFERENCE_IMG_FILE_NAME,
        IMG_DB_NAME,
        IMG_DIR_NAME,
        current_cropped_faces_dir_in_cropped_img_db_path,
    )

    if most_similar_cropped_img_path:
        # set destination path (for copying photo)
        new_most_similar_cropped_img_path = os.path.join(
            cropped_img_db_name, IMG_DIR_NAME, "cropped_" + IMG_FILE_NAME
        )

        # Load most similar cropped image from directory (copy)
        most_similar_cropped_img = load_image(most_similar_cropped_img_path)

        # Then save (paste)
        save_image(new_most_similar_cropped_img_path, most_similar_cropped_img)

logging.info("Code executed without crashing.")
