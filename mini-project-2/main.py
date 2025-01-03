import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to perform convolution over an image, with the given kernel
def convolution(im, kernel):
    kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    im_height, im_width = im.shape
    kernel_size = kernel.shape[0]
    pad_size = int((kernel_size - 1) / 2)
    im_padded = np.zeros((im_height + pad_size * 2, im_width + pad_size * 2))
    im_padded[pad_size:-pad_size, pad_size:-pad_size] = im

    im_out = np.zeros_like(im)
    for x in range(im_width):
        for y in range(im_height):
            im_patch = im_padded[y:y + kernel_size, x:x + kernel_size]
            new_value = np.sum(kernel * im_patch)
            im_out[y, x] = new_value
    return im_out


# Function to retrieve a Gaussian kernel of the specified size
def get_gaussian_kernel(kernel_size, sigma):
    kernel_x = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    for i in range(kernel_size):
        kernel_x[i] = np.exp(-(kernel_x[i] / sigma) ** 2 / 2)
    kernel = np.outer(kernel_x.T, kernel_x.T)

    kernel *= 1.0 / kernel.sum()
    return kernel


# Function to compute the gradient of an image, returning the magnitude and direction for each pixel
def compute_gradient(im):
    sobel_filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gradient_x = convolution(im, sobel_filter_x)
    gradient_y = convolution(im, sobel_filter_y)

    magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    magnitude *= 255.0 / magnitude.max()
    direction = np.arctan2(gradient_y, gradient_x)
    direction *= 180 / np.pi
    return magnitude, direction


# Function to perform non-maximum suppression given the gradient magnitude and direction
def nms(magnitude, direction):
    height, width = magnitude.shape
    res = np.zeros(magnitude.shape)
    direction[direction < 0] += 180  # (-180, 180) -> (0, 180)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            current_direction = direction[y, x]
            current_magnitude = magnitude[y, x]
            if (0 <= current_direction < 22.5) or (157.5 <= current_direction <= 180):
                p = magnitude[y, x - 1]
                r = magnitude[y, x + 1]

            elif 22.5 <= current_direction < 67.5:
                p = magnitude[y + 1, x + 1]
                r = magnitude[y - 1, x - 1]

            elif 67.5 <= current_direction < 112.5:
                p = magnitude[y - 1, x]
                r = magnitude[y + 1, x]

            else:
                p = magnitude[y - 1, x + 1]
                r = magnitude[y + 1, x - 1]

            if current_magnitude >= p and current_magnitude >= r:
                res[y, x] = current_magnitude

    return res


# Function to perform Hysteresis Thresholding by keeping edges that are strong. It doesn't check for
# weak edges connected to strong edges to make these weak edges "strong"
def hysteresis_thresholding(im, low_threshold, high_threshold):
    strong_edge_val = 255  # value for strong edges
    strong_edges = np.zeros_like(im)
    # weak_edge_val = 50  # value for weak edges
    # weak_edges = np.zeros_like(im)

    strong_y, strong_x = np.where(im > high_threshold)  # setting for strong edges, getting indices
    strong_edges[strong_y, strong_x] = strong_edge_val  # write strong edges

    # weak_y, weak_x = np.where((im >= low_threshold) & (im <= high_threshold))  # setting for weak edges
    # weak_edges[weak_y, weak_x] = weak_edge_val  # write weak edges

    return strong_edges


# Function to perform Hough Transform over a map with edges, or a black and white image
# It returns an accumulator array, theta values and rho values
def hough_transform(edge_map):
    theta_values = np.deg2rad(np.arange(-90.0, 90.0))
    height, width = edge_map.shape
    diagonal_length = int(round(math.sqrt(width * width + height * height)))
    rho_values = np.linspace(-diagonal_length, diagonal_length, diagonal_length * 2 + 1)

    accumulator = np.zeros((len(rho_values), len(theta_values)), dtype=int)
    y_coordinates, x_coordinates = np.nonzero(edge_map)

    for edge_idx in range(len(x_coordinates)):
        x = x_coordinates[edge_idx]
        y = y_coordinates[edge_idx]
        for theta_idx in range(len(theta_values)):
            theta = theta_values[theta_idx]
            rho = int(round(x * np.cos(theta) + y * np.sin(theta)))
            accumulator[rho + diagonal_length, theta_idx] += 1

    return accumulator, theta_values, rho_values


# Function to perform non-maximum suppression given the accumulator array from a Hough Transform
# operation. Edges above the threshold are considered for the suppressed accumulator, and
# the window size indicates size of the local window used to find a local maximum
def nms_hough(accumulator, threshold=50, window_size=3):
    height, width = accumulator.shape
    suppressed_accumulator = np.zeros_like(accumulator)

    half_window = window_size // 2  # window size, used to calculate local maximum

    for r in range(half_window, height - half_window):
        for t in range(half_window, width - half_window):
            window = accumulator[r - half_window:r + half_window + 1, t - half_window:t + half_window + 1]  # getting window
            local_max = np.max(window)  # find local maximum

            # current point is local maximum and above threshold
            if accumulator[r, t] == local_max and accumulator[r, t] > threshold:
                suppressed_accumulator[r, t] = accumulator[r, t]

    return suppressed_accumulator


# Function used to draw the lines in the image
def draw_hough_image(im, suppressed_accumulator, rho_values, theta_values):
    lines = np.argwhere(suppressed_accumulator)     # get lines
    height, width = im.shape[:2]

    # draw lines in the image
    for line in lines:
        rho = rho_values[line[0]]
        theta = theta_values[line[1]]
        slope = -np.cos(theta) / np.sin(theta)
        intercept = rho / np.sin(theta)
        x1, x2 = 0, width
        y1 = int(slope * x1 + intercept)
        y2 = int(slope * x2 + intercept)
        cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return im


# Function to perform Hough Transformation using the cv2 library (cv2.HoughLines), and write the lines into the image
def cv2_hough_transform(im, edge_threshold=70):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 70, 150)     # get edges from gray image using Canny detector

    lines = cv2.HoughLines(edges, 1, np.pi/180, edge_threshold)     # get lines

    # draw the lines in the image
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return im


# Run this function to obtain all the output images with different kernel sizes for the homework.
def main():
    # Performing Hysteresis Thresholding in Canny edge on lena.png
    im_lena = cv2.imread("lena.png", 0)
    im_lena = im_lena.astype(float)

    im_smoothed = convolution(im_lena, get_gaussian_kernel(9, 3))   # get smoothed image

    gradient_magnitude, gradient_direction = compute_gradient(im_smoothed)  # get gradient magnitude and direction

    edge_nms = nms(gradient_magnitude, gradient_direction)      # perform non-maximum suppression

    cv2.imwrite("lena_edge_gradient.png", gradient_magnitude.astype(np.uint8))  # write gradient
    cv2.imwrite("lena_edge_NMS.png", edge_nms.astype(np.uint8))     # write gradient after NMS

    edges_hysteresis = hysteresis_thresholding(edge_nms, 10, 30)    # low: 10, high: 30
    cv2.imwrite("lena_edges_hysteresis_10_30.png", edges_hysteresis.astype(np.uint8))

    edges_hysteresis = hysteresis_thresholding(edge_nms, 20, 40)    # low: 20, high: 40
    cv2.imwrite("lena_edges_hysteresis_20_40.png", edges_hysteresis.astype(np.uint8))

    edges_hysteresis = hysteresis_thresholding(edge_nms, 30, 50)    # low: 30, high: 50
    cv2.imwrite("lena_edges_hysteresis_30_50.png", edges_hysteresis.astype(np.uint8))

    # Using cv2.Canny to process lena.png
    canny_lena = cv2.Canny(cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE), 70, 150)
    cv2.imwrite("lena_edges_canny_cv2.png", canny_lena)

    # Performing Hough Transform on paper.bmp and shape.bmp
    im_paper = cv2.imread('paper.bmp')
    im_shape = cv2.imread('shape.bmp')

    # Hough Transform on paper.bmp
    edge_map = cv2.Canny(cv2.cvtColor(im_paper, cv2.COLOR_BGR2GRAY), 70, 150)   # get edges with Canny

    accumulator, theta_values, rho_values = hough_transform(edge_map)  # calculate Hough Transform

    suppressed_accumulator = nms_hough(accumulator)  # use non-maximum suppression for the accumulator

    im_output = draw_hough_image(im_paper, suppressed_accumulator, rho_values, theta_values)

    cv2.imwrite("paper_accumulator.png", (accumulator * 255 / accumulator.max()).astype(np.uint8))
    cv2.imwrite("paper_accumulator_suppressed.png",
                (suppressed_accumulator * 255 / suppressed_accumulator.max()).astype(np.uint8))
    cv2.imwrite("paper_hough_transform_output.png", im_output)

    # Using cv2 to apply Hough Transform for paper.bmp
    hough_lines_cv2 = cv2_hough_transform(im_paper)
    cv2.imwrite("paper_hough_transform_cv2.png", hough_lines_cv2)

    # Hough Transform on shape.bmp
    im_gray = cv2.cvtColor(im_shape, cv2.COLOR_BGR2GRAY)
    edge_map = cv2.Canny(im_gray, 70, 150)

    accumulator, theta_values, rho_values = hough_transform(edge_map)  # calculate Hough Transform

    suppressed_accumulator = nms_hough(accumulator)  # use non-maximum suppression for the accumulator

    im_output = draw_hough_image(im_shape, suppressed_accumulator, rho_values, theta_values)

    cv2.imwrite("shape_accumulator.png", (accumulator * 255 / accumulator.max()).astype(np.uint8))
    cv2.imwrite("shape_accumulator_suppressed.png",
                (suppressed_accumulator * 255 / suppressed_accumulator.max()).astype(np.uint8))
    cv2.imwrite("shape_hough_transform_output.png", im_output)

    # Using cv2 to apply Hough Transform for shape.bmp
    hough_lines_cv2 = cv2_hough_transform(im_shape)
    cv2.imwrite("shape_hough_transform_cv2.png", hough_lines_cv2)


if __name__ == "__main__":
    main()
