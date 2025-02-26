import os
import cv2
from utils import get_project_dir, get_directory_paths, print_info, IMAGES_PER_CLASS


def create_img():
    # Get project directory and data directory from utils
    PROJECT_DIR = get_project_dir()
    DATA_DIR = get_directory_paths()["data"]

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    number_of_classes = 26
    target_images = IMAGES_PER_CLASS

    cap = cv2.VideoCapture(0)
    # Loop over each class
    for j in range(number_of_classes):
        class_dir = os.path.join(DATA_DIR, str(j))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        print_info(f"Collecting data for class {j}")

        # Wait for user confirmation to start capturing images for the current class
        while True:
            ret, frame = cap.read()
            cv2.putText(
                frame,
                'Ready? Press "Q" to start for class "{}"'.format(j),
                (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.imshow("frame", frame)
            if cv2.waitKey(25) == ord("q"):
                break

        # Capture training images for current class
        counter = 0
        while counter < target_images:
            ret, frame = cap.read()
            cv2.imshow("frame", frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(class_dir, f"{counter}.jpg"), frame)
            counter += 1

    cap.release()
    cv2.destroyAllWindows()


# Allow the module to be imported without running the script and run only when executed directly
if __name__ == "__main__":
    create_img()
