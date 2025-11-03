"""
IKT213 - Assignment 4, Part 1


Task:
Perform Harris Corner Detection on the reference image and save the output.
"""

import cv2
import numpy as np

def harris_corner_detection(reference_image_path, output_path="harris.png"):
    img = cv2.imread(reference_image_path)
    if img is None:
        raise FileNotFoundError("Reference image not found.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)

    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imwrite(output_path, img)
    print(f"[INFO] Harris corners saved as {output_path}")

if __name__ == "__main__":
    harris_corner_detection("reference_img.png")
    print("completed successfully.")
