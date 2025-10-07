import cv2
import numpy as np
import os

# Make sure output folder exists
os.makedirs("output_images", exist_ok=True)

# Load the image
image = cv2.imread("lena.png")


# 1. Padding

def padding(image, border_width):
    padded_img = cv2.copyMakeBorder(
        image,
        border_width, border_width, border_width, border_width,
        cv2.BORDER_REFLECT
    )
    cv2.imwrite("output_images/padded_image.png", padded_img)
    return padded_img



# 2. Cropping

def crop(image, x_0, x_1, y_0, y_1):
    cropped_img = image[y_0:y_1, x_0:x_1]
    cv2.imwrite("output_images/cropped_image.png", cropped_img)
    return cropped_img



# 3. Resize

def resize(image, width, height):
    resized_img = cv2.resize(image, (width, height))
    cv2.imwrite("output_images/resized_image.png", resized_img)
    return resized_img



# 4. Manual Copy

def copy(image, emptyPictureArray):
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            emptyPictureArray[y, x] = image[y, x]
    cv2.imwrite("output_images/copied_image.png", emptyPictureArray)
    return emptyPictureArray



# 5. Grayscale

def grayscale(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("output_images/grayscale_image.png", gray_img)
    return gray_img



# 6. HSV

def hsv(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite("output_images/hsv_image.png", hsv_img)
    return hsv_img



# 7. Hue Shift

def hue_shifted(image, emptyPictureArray, hue):
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                new_val = int(image[y, x, c]) + hue
                emptyPictureArray[y, x, c] = np.clip(new_val, 0, 255)
    cv2.imwrite("output_images/hue_shifted_image.png", emptyPictureArray)
    return emptyPictureArray



# 8. Smoothing (Gaussian Blur)

def smoothing(image):
    blurred_img = cv2.GaussianBlur(image, (15, 15), 0)
    cv2.imwrite("output_images/smoothed_image.png", blurred_img)
    return blurred_img



# 9. Rotation

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated_img = cv2.rotate(image, cv2.ROTATE_180)
    else:
        raise ValueError("Rotation angle must be 90 or 180 degrees.")
    cv2.imwrite(f"output_images/rotated_{rotation_angle}.png", rotated_img)
    return rotated_img



# Run all functions

if __name__ == "__main__":
    print("Running image processing functions...")

    padded = padding(image, 100)
    cropped = crop(image, 80, image.shape[1] - 130, 80, image.shape[0] - 130)
    resized = resize(image, 200, 200)

    empty_array = np.zeros_like(image, dtype=np.uint8)
    copied = copy(image, empty_array.copy())

    gray = grayscale(image)
    hsv_img = hsv(image)

    hue_array = np.zeros_like(image, dtype=np.uint8)
    hue_shift = hue_shifted(image, hue_array.copy(), 50)

    smooth = smoothing(image)
    rotate_90 = rotation(image, 90)
    rotate_180 = rotation(image, 180)

    print("All images processed and saved in the 'output_images' folder.")
