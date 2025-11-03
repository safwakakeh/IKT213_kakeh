"""
IKT213 - Assignment 4, Part 1

Task:
Perform Harris Corner Detection on the reference image,
save the output image, and create a one-page PDF with the result.
"""

import cv2
import numpy as np
import os
from fpdf import FPDF


def harris_corner_detection(reference_image_path, output_folder="output", image_name="harris.png"):
    """
    Detects Harris corners on the given reference image and saves the result.
    Also generates a one-page PDF with the output image.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Read input image
    img = cv2.imread(reference_image_path)
    if img is None:
        raise FileNotFoundError("Reference image not found. Please check the file path.")

    # Convert to grayscale and float32
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Apply Harris Corner Detection
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)

    # Mark corners in red
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    # Save the result image
    image_path = os.path.join(output_folder, image_name)
    cv2.imwrite(image_path, img)
    print(f"[INFO] Harris corners image saved at: {image_path}")

    # Create PDF with the result image
    pdf_name = "assignment_4_part1_harris.pdf"
    pdf_path = os.path.join(output_folder, pdf_name)
    pdf = FPDF(unit="mm", format="A4")
    pdf.add_page()

    # Add title text on the page
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Part 1 - Harris Corner Detection", ln=True, align="C")

    # Insert the image
    pdf.image(image_path, x=10, y=25, w=190)
    pdf.output(pdf_path)

    print(f"[INFO] PDF file created at: {pdf_path}")
    print("completed successfully.")
    return image_path, pdf_path


if __name__ == "__main__":
    # Run Harris corner detection on reference image
    harris_corner_detection("reference_img-1.png")
