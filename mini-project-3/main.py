import numpy as np
import cv2
import math
import glob
from scipy.stats import multivariate_normal

# Calculate HS histogram for a single image
def calculate_hs_histogram(img, bin_size):
    height, width, _ = img.shape
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    max_h = 180
    max_s = 256
    num_bins_h = math.ceil(max_h / bin_size)
    num_bins_s = math.ceil(max_s / bin_size)
    hs_hist = np.zeros((num_bins_h, num_bins_s))
    for i in range(height):
        for j in range(width):
            h = img_hsv[i, j, 0]
            s = img_hsv[i, j, 1]
            hs_hist[math.floor(h / bin_size), math.floor(s / bin_size)] += 1
    hs_hist /= hs_hist.sum()
    return hs_hist

# Calculate HS histogram for multiple images and find average
def calculate_average_hs_histogram(images, bin_size):
    max_h = 180
    max_s = 256
    num_bins_h = math.ceil(max_h / bin_size)
    num_bins_s = math.ceil(max_s / bin_size)
    avg_hist = np.zeros((num_bins_h, num_bins_s))
    for img_path in images:
        img = cv2.imread(img_path)
        if img is not None:
            hs_hist = calculate_hs_histogram(img, bin_size)
            avg_hist += hs_hist
    avg_hist /= len(images)  # Average over all images
    return avg_hist

# Perform color segmentation on an image based on a histogram
def color_segmentation(img, hs_hist, bin_size, threshold):
    height, width, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros((height, width, 1), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            h = hsv[i, j, 0]
            s = hsv[i, j, 1]
            if hs_hist[math.floor(h / bin_size), math.floor(s / bin_size)] > threshold:
                mask[i, j, 0] = 1
    return mask

# Calculate Gaussian mean and covariance matrix from images
def calculate_gaussian(images):
    hs_values = []

    for img_path in images:     # Collect HS values from all images
        img = cv2.imread(img_path)
        if img is not None:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h = img_hsv[:, :, 0].flatten()
            s = img_hsv[:, :, 1].flatten()
            hs_values.extend(zip(h, s))

    hs_values = np.array(hs_values)
    mean_vector = np.mean(hs_values, axis=0)
    covariance_matrix = np.cov(hs_values, rowvar=False)

    return mean_vector, covariance_matrix

# Perform color segmentation on an image based on Gaussian skin color probability model
def gaussian_color_segmentation(img, mean, cov, threshold):
    height, width, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros((height, width), dtype=np.uint8)

    gaussian_model = multivariate_normal(mean=mean, cov=cov)

    for i in range(height):
        for j in range(width):
            h, s, _ = hsv[i, j]
            hs_pixel = np.array([h, s])
            prob = gaussian_model.pdf(hs_pixel)  # Calculate skin probability
            if prob > threshold:
                mask[i, j] = 255  # Mark as skin in the mask

    return mask

# Perform Harris corner detection using OpenCV
def harris_corner_detection(image_path, block_size=2, ksize=5, k=0.07, threshold=0.01):

    img = cv2.imread(image_path)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_image = np.float32(gray_image)

    dst = cv2.cornerHarris(gray_image, block_size, ksize, k)

    dst = cv2.dilate(dst, None)

    img[dst > threshold * dst.max()] = [0, 0, 255]

    return img

# For histogram
# Training
skin_patch_images = glob.glob("skin-images/*.png")
bin_size = 20
hs_hist = calculate_average_hs_histogram(skin_patch_images, bin_size)

# Testing
img_test = cv2.imread("testing_image.bmp")
threshold = 0.03
mask = color_segmentation(img_test, hs_hist, bin_size, threshold)

img_seg = img_test * mask

# Save the results
cv2.imwrite("hs_input.png", img_test)
cv2.imwrite("hs_mask.png", (mask * 255).astype(np.uint8))
cv2.imwrite("hs_segmentation_after.png", img_seg.astype(np.uint8))

# For Gaussian Model
# Training
mean_vector, covariance_matrix = calculate_gaussian(skin_patch_images)

# Testing
img_test = cv2.imread("testing_image.bmp")
threshold = 1e-4    # Threshold for Gaussian
mask = gaussian_color_segmentation(img_test, mean_vector, covariance_matrix, threshold)

# Apply mask to the original image for segmented result
img_seg = cv2.bitwise_and(img_test, img_test, mask=mask)

# Save the results
cv2.imwrite("gaussian_input.png", img_test)
cv2.imwrite("gaussian_mask.png", mask)
cv2.imwrite("gaussian_segmentation_after.png", img_seg)

# For Harris corner detector
checkerboard_image = harris_corner_detection('checkerboard.png')

toy_image = harris_corner_detection('toy.png')

cv2.imwrite('harris_corner_checkerboard.png', checkerboard_image)
cv2.imwrite('harris_corner_toy.png', toy_image)

