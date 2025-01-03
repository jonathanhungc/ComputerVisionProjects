import cv2
import numpy as np


# Helper function to apply rotation to a point
def rotate_point(point, angle, center=np.array([0, 0])):
    # Convert angle to radians
    theta = np.radians(angle)

    # Create the rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    rotated_point = rotation_matrix @ (point - center) + center
    return rotated_point


# Helper function to draw a circle
def draw_circle(image, point, color, radius=10):
    cv2.circle(image, tuple(point.astype(int)), radius=radius, color=color, thickness=-1)


# Helper function to apply an affine transformation
def apply_transformation(image, matrix, output_filename):
    transformed_image = cv2.warpAffine(image, matrix, (w, h))
    cv2.imwrite(output_filename, transformed_image)


# P1
image = np.ones((500, 500, 3), dtype=np.uint8) * 255

# Coordinates for points a and c
a = np.array([100, 40])
c = np.array([100, 100])

# Draw red circle at point a
draw_circle(image, a, color=(0, 0, 255))

# Rotate a around by 60 degrees clockwise
b = rotate_point(a, 60)
draw_circle(image, b, color=(0, 255, 0))  # Green circle at b

# Draw the black circle at point c
draw_circle(image, c, color=(0, 0, 0))

# Rotate a around c by 60 degrees clockwise to get d
d = rotate_point(a, 60, center=c)
draw_circle(image, d, color=(255, 0, 0))  # Blue circle at d

cv2.imwrite("transformed_image.png", image)

# P2
image = cv2.imread('lena.png')

h, w = image.shape[:2]
center_x, center_y = w // 2, h // 2

# Move right by 100 pixels and down by 200 pixels
move_matrix = np.float32([[1, 0, 100], [0, 1, 200]])
apply_transformation(image, move_matrix, "lena_move.png")

# Horizontal flip about the image center
flip_matrix = np.float32([[-1, 0, w], [0, 1, 0]])
apply_transformation(image, flip_matrix, "lena_flip.png")

# Rotation about the origin by 45 degrees clockwise
theta = np.radians(45)
rotation_matrix_origin = np.float32([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0]
])
apply_transformation(image, rotation_matrix_origin, "lena_rotate_origin.png")

# Rotation about the image center by 45 degrees clockwise
rotation_matrix_center = np.float32([
    [np.cos(theta), -np.sin(theta), center_x - center_x * np.cos(theta) + center_y * np.sin(theta)],
    [np.sin(theta), np.cos(theta), center_y - center_x * np.sin(theta) - center_y * np.cos(theta)]
])
apply_transformation(image, rotation_matrix_center, "lena_rotate_center.png")