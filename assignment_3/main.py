import cv2
import numpy as np


# 1. Sobel edge detection
def sobel_edge_detection(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # dx=1, dy=1, ksize=1
    sobel = cv2.Sobel(blurred, cv2.CV_64F, 1, 1, ksize=1)
    sobel = cv2.convertScaleAbs(sobel)

    cv2.imwrite("output/sobel_edges.png", sobel)
    print("Sobel edge image saved as sobel_edges.png")
    return sobel


# 2. Canny edge detection
def canny_edge_detection(image_path, threshold_1=50, threshold_2=50):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(blurred, threshold_1, threshold_2)

    cv2.imwrite("output/canny_edges.png", edges)
    print("Canny edge image saved as canny_edges.png")
    return edges


# 3. Template matching
def template_match(image_path, template_path):
    image = cv2.imread(image_path)
    template = cv2.imread(template_path)

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(result >= threshold)

    h, w = gray_template.shape
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite("output/template_matched.png", image)
    print("Template matching result saved as template_matched.png")
    return image


# 4. Resizing (zoom in/out)
def resize(image_path, scale_factor=2, up_or_down="up"):
    import os
    import cv2

    # Leser inn bildet
    image = cv2.imread(image_path)

    # Sjekker at output-mappen finnes, ellers lager den 
    os.makedirs("output", exist_ok=True)

    # Zoom inn eller ut
    if up_or_down == "up":
        resized = cv2.pyrUp(image, dstsize=(image.shape[1] * scale_factor, image.shape[0] * scale_factor))
    elif up_or_down == "down":
        resized = cv2.pyrDown(image)
    else:
        raise ValueError("up_or_down must be 'up' or 'down'")

    # Lagre bildet i output-mappen
    output_path = f"output/resized_{up_or_down}.png"
    cv2.imwrite(output_path, resized)
    print(f"Image resized {up_or_down} saved as {output_path}")

    return resized



#  Test
if __name__ == "__main__":
    sobel_edge_detection("lambo.png")
    canny_edge_detection("lambo.png")
    template_match("shapes.png", "shapes_template.jpg")
    resize("lambo.png", scale_factor=2, up_or_down="up")
    resize("lambo.png", scale_factor=2, up_or_down="down")
