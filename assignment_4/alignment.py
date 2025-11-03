"""
IKT213 - Part 2


Task:
Align a target image with the reference image using SIFT feature matching.
( i want to sey that i tryed fift matching but it´s feiled so  i tryed to go to chosse to and used ORB matching )
 I Save aligned and match visualization images in the output folder,
and create a combined PDF file for submission.
"""

import cv2
import numpy as np
from fpdf import FPDF
import os


def align_images_sift(image_to_align_path, reference_image_path,
                      max_features=500, good_match_percent=0.9,
                      output_folder="output"):
    """
    Aligns image_to_align to reference_image using SIFT and homography.
    Saves aligned and match visualization results to the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    im1 = cv2.imread(image_to_align_path, cv2.IMREAD_COLOR)
    im2 = cv2.imread(reference_image_path, cv2.IMREAD_COLOR)
    if im1 is None or im2 is None:
        raise FileNotFoundError("Make sure both images are in the assignment_4 folder.")

    # Convert to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect SIFT features and descriptors
    sift = cv2.SIFT_create(max_features)
    keypoints1, descriptors1 = sift.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(im2_gray, None)

    print(f"[DEBUG] Keypoints in image_to_align: {len(keypoints1)}")
    print(f"[DEBUG] Keypoints in reference_image: {len(keypoints2)}")

    # Match features using FLANN
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < good_match_percent * n.distance]
    print(f"[INFO] Number of good matches (SIFT): {len(good_matches)}")

    # Draw matches
    matches_path = os.path.join(output_folder, "matches.png")
    match_img = cv2.drawMatches(im1, keypoints1, im2, keypoints2, good_matches, None)
    cv2.imwrite(matches_path, match_img)

    # If not enough matches, switch to ORB automatically
    if len(good_matches) < 4:
        print("[WARN] Not enough SIFT matches found. Trying ORB instead...")
        return align_images_orb(image_to_align_path, reference_image_path, output_folder)

    # Extract matched keypoints
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    # Compute homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    height, width, channels = im2.shape
    aligned = cv2.warpPerspective(im1, h, (width, height))

    aligned_path = os.path.join(output_folder, "aligned.png")
    cv2.imwrite(aligned_path, aligned)
    print(f"[INFO] Saved aligned image to {aligned_path}")
    print(f"[INFO] Saved match visualization to {matches_path}")
    return True


def align_images_orb(image_to_align_path, reference_image_path, output_folder="output"):
    """
    Aligns the two images using ORB feature matching (used if SIFT fails).
    """
    os.makedirs(output_folder, exist_ok=True)
    im1 = cv2.imread(image_to_align_path, cv2.IMREAD_COLOR)
    im2 = cv2.imread(reference_image_path, cv2.IMREAD_COLOR)

    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(1500)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    print(f"[DEBUG] ORB keypoints: {len(keypoints1)} vs {len(keypoints2)}")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    num_good = int(len(matches) * 0.15)
    good = matches[:max(num_good, 4)]
    print(f"[INFO] Number of good matches (ORB): {len(good)}")

    matches_path = os.path.join(output_folder, "matches_orb.png")
    match_img = cv2.drawMatches(im1, keypoints1, im2, keypoints2, good, None)
    cv2.imwrite(matches_path, match_img)

    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 2)

    h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    height, width, _ = im2.shape
    aligned = cv2.warpPerspective(im1, h, (width, height))
    aligned_path = os.path.join(output_folder, "aligned_orb.png")
    cv2.imwrite(aligned_path, aligned)
    print(f"[INFO] ORB alignment saved to {aligned_path}")
    return True


def make_pdf(output_folder="output", pdf_name="assignment_4_output.pdf"):
    """
    Combines the output images into a single multi-page PDF.
    """
    pdf = FPDF(unit="mm", format="A4")
    image_list = ["harris.png", "aligned.png", "matches.png"]

    for image in image_list:
        path = os.path.join(output_folder, image)
        if not os.path.exists(path):
            print(f"[WARN] Missing image: {path}, skipping.")
            continue
        pdf.add_page()
        pdf.image(path, x=10, y=10, w=190)

    output_path = os.path.join(output_folder, pdf_name)
    pdf.output(output_path, "F")
    print(f"[INFO] PDF file saved as {output_path}")


if __name__ == "__main__":
    align_images_sift("align_this.jpg", "reference_img.png",
                      max_features=500, good_match_percent=0.9,
                      output_folder="output")

    make_pdf(output_folder="output")
    print("completed successfully.")
