import cv2
import os


def print_image_information(image):
    """Task IV: Print information about an image"""
    height, width, channels = image.shape
    print(f"Height: {height}")
    print(f"Width: {width}")
    print(f"Channels: {channels}")
    print(f"Size: {image.size}")
    print(f"Data type: {image.dtype}")


def save_camera_information(output_path):
    """Task V: Read from the webcam and save info to a txt file"""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"fps: {fps:.2f}\n")
        f.write(f"width: {int(width)}\n")
        f.write(f"height: {int(height)}\n")

    cap.release()
    print(f"Camera info saved to {output_path}")


def main():
    # Task IV
    image = cv2.imread("lena.png")
    if image is None:
        print("Error: Could not find lena.png in the project folder.")
    else:
        print_image_information(image)

    # Task V
    output_file = os.path.expanduser("~/IKT213_kakeh/assignment_1/solutions/camera_outputs.txt")
    save_camera_information(output_file)


if __name__ == "__main__":
    main()
