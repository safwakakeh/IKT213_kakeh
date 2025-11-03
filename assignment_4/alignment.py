"""
IKT213 - Part 2 (ORB Only)


Task:
Align 'align_this.jpg' with 'reference_img-1.png' using only ORB feature matching.

I allredy tryed sift maching , but it was not inof maching , so that wry i tryed ORB to get better result.
 I Save aligned and match visualization images in the output folder,
and create a combined PDF file for submission.
"""

import cv2
import numpy as np
import os
from fpdf import FPDF


def align_images_orb(image_to_align_path, reference_image_path,
                     max_features=2000, good_match_percent=0.15,
                     output_folder="output"):
    """
    Aligns image_to_align to reference_image using ORB and homography.
    Saves aligned and match visualization results to the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    print("\n[INFO] Starting ORB alignment...")

    # Read both images
    im1 = cv2.imread(image_to_align_path, cv2.IMREAD_COLOR)
    im2 = cv2.imread(reference_image_path, cv2.IMREAD_COLOR)
    if im1 is None or im2 is None:
        raise FileNotFoundError(" One or both images not found. Check file names and paths.")

    # Convert to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    print(f"[DEBUG] ORB keypoints: align_this={len(keypoints1)}, reference={len(keypoints2)}")

    # Match features using Brute-Force matcher with Hamming distance
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Select top percentage of matches
    num_good_matches = int(len(matches) * good_match_percent)
    good_matches = matches[:max(4, num_good_matches)]
    print(f"[INFO] Good matches: {len(good_matches)}")

    # Draw matches for visualization
    matches_path = os.path.join(output_folder, "matches_orb.png")
    matches_img = cv2.drawMatches(im1, keypoints1, im2, keypoints2, good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(matches_path, matches_img)
    print(f"[INFO] Saved match visualization: {matches_path}")

    if len(good_matches) < 4:
        raise RuntimeError(" Not enough good matches to compute homography (need at least 4).")

    # Extract matched keypoints
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography matrix using RANSAC
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError(" Homography computation failed.")

    # Warp the image to align it with the reference
    height, width, channels = im2.shape
    aligned = cv2.warpPerspective(im1, H, (width, height))

    aligned_path = os.path.join(output_folder, "aligned_orb.png")
    cv2.imwrite(aligned_path, aligned)
    print(f"[INFO] Saved aligned image: {aligned_path}")

    print("\n ORB alignment completed successfully!")
    return aligned_path, matches_path


def make_pdf(output_folder="output", pdf_name="assignment_4_output_orb.pdf"):
    """
    Combine output images into a single PDF file for submission.
    """
    pdf = FPDF(unit="mm", format="A4")
    image_list = ["harris.png", "aligned_orb.png", "matches_orb.png"]

    for image in image_list:
        path = os.path.join(output_folder, image)
        if not os.path.exists(path):
            print(f"[WARN] Skipping missing image: {path}")
            continue
        pdf.add_page()
        pdf.image(path, x=10, y=10, w=190)

    output_path = os.path.join(output_folder, pdf_name)
    pdf.output(output_path)
    print(f"[INFO] PDF created: {output_path}")


if __name__ == "__main__":
    align_images_orb("align_this.jpg", "reference_img-1.png",
                     max_features=2000, good_match_percent=0.15,
                     output_folder="output")

    make_pdf(output_folder="output")
    print("\n (ORB only) finished successfully.")
