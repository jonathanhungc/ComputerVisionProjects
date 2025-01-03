import sys
import cv2
import numpy as np


# Functions to perform mean filter, input filename and kernel size
def mean_filter(filename, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=int) / (kernel_size * kernel_size)

    convolution(filename, "mean", kernel)


# Functions to perform gaussian filter, input filename, kernel size and sigma
def gaussian_filter(filename, kernel_size, sigma):

    x_dir = kernel_size // 2
    y_dir = kernel_size // 2

    kernel_gaussian = np.zeros((kernel_size, kernel_size), dtype=float)     # for the gaussian kernel

    for x in range(-x_dir, x_dir + 1):
        for y in range(-y_dir, y_dir + 1):
            exp_term = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            kernel_gaussian[x + x_dir, y + y_dir] = (1 / (2 * np.pi * sigma ** 2)) * exp_term

    #print(kernel)
    convolution(filename, "gaussian", kernel_gaussian)


# Functions to perform sharpening filter, input filename and kernel size
def sharpening_filter(filename, kernel_size):
    kernel_mean = np.ones((kernel_size, kernel_size), dtype=int) / (kernel_size * kernel_size)
    #print(kernel_mean)

    matrix = np.zeros((kernel_size, kernel_size), dtype=int)
    matrix[kernel_size // 2, kernel_size // 2] = 1

    convolution(filename, "sharpening", (2*matrix - kernel_mean))


# Functions to perform convolution, input filename, name of the filter applied, and kernel
def convolution(filename, filter_name, kernel):
    im = cv2.imread(filename + ".png")
    im = im.astype(float)

    im_height, im_width, im_channels = im.shape  # number of rows, columns and color channels
    kernel_size = kernel.shape[0]
    pad_size = int((kernel_size - 1) / 2)  # padding based on kernel size
    im_padded = np.zeros((im_height + pad_size * 2, im_width + pad_size * 2, im_channels))
    im_padded[pad_size:-pad_size, pad_size:-pad_size, :] = im  # adding padding to image

    im_out = np.zeros_like(im)
    for c in range(im_channels):
        for x in range(im_width):
            for y in range(im_height):
                im_patch = im_padded[y:y + kernel_size, x:x + kernel_size, c]
                new_value = np.sum(kernel * im_patch)   # perform linear combination between kernel and image patch
                im_out[y, x, c] = new_value

    im_out = np.clip(im_out, 0, 255)    # clip values out of boundaries
    im_out = im_out.astype(np.uint8)
    cv2.imwrite(filename + "_" + filter_name + "_" + str(kernel_size) + "x" + str(kernel_size) + ".png", im_out)


# Functions to perform median filter, input filename and kernel size. Very similar to convolution method.
# I could have combined both and just apply a switch for when to apply convolution and when to apply median filter,
# but I thought it looked clearer by splitting them.

def median_filter(filename, kernel_size):
    im = cv2.imread(filename + ".png")
    im = im.astype(float)

    im_height, im_width, im_channels = im.shape  # number of rows, columns and color channels
    pad_size = int((kernel_size - 1) / 2)  # padding based on kernel size
    im_padded = np.zeros((im_height + pad_size * 2, im_width + pad_size * 2, im_channels))
    im_padded[pad_size:-pad_size, pad_size:-pad_size, :] = im  # adding padding to image

    im_out = np.zeros_like(im)
    for c in range(im_channels):
        for x in range(im_width):
            for y in range(im_height):
                im_patch = im_padded[y:y + kernel_size, x:x + kernel_size, c]
                new_value = np.median(im_patch)     # find the median of the image patch
                im_out[y, x, c] = new_value

    im_out = np.clip(im_out, 0, 255)     # clip values out of boundaries
    im_out = im_out.astype(np.uint8)
    cv2.imwrite(filename + "_median_" + str(kernel_size) + "x" + str(kernel_size) + ".png", im_out)


# Functions to perform gaussian filter using filter2D, input filename, kernel size, and sigma
def gaussian_opencv(filename, kernel_size, sigma):
    im = cv2.imread(filename + ".png")
    im = im.astype(float)

    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = kernel @ kernel.T

    im_out = cv2.filter2D(im, -1, kernel)

    cv2.imwrite(filename + "_gaussian_opencv_" + str(kernel_size) + "x" + str(kernel_size) + ".png", im_out)


# Run this function to obtain all the output images with different kernel sizes for the homework.
def main():
    # P1 lena.png images
    for i in [3, 5, 7]:
        mean_filter("lena", i)
        gaussian_filter("lena", i, 1)
        sharpening_filter("lena", i)

    # P2 art.png images
    for i in [3, 5, 7, 9]:
        mean_filter("art", i)
        median_filter("art", i)

    # P3 lena.png using filter2D
    for i in [3, 5, 7]:
        gaussian_opencv("lena", i, 1)


if __name__ == "__main__":
    main()
