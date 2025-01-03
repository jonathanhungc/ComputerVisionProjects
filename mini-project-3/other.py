import numpy as np

# Define a sample 5x5 grayscale image
image = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 0, 2, 1],
    [1, 2, 2, 1, 1],
    [1, 2, 1, 0, 1],
    [1, 1, 1, 1, 1]
])


def compute_hessian_matrix(window):
    """
    Computes the Hessian matrix at the center pixel of a 3x3 window.
    """
    # Finite difference approximations
    Ix = window[2, 1] - window[0, 1]
    Iy = window[1, 2] - window[1, 0]

    Ixx = window[2, 1] - 2 * window[1, 1] + window[0, 1]
    Iyy = window[1, 2] - 2 * window[1, 1] + window[1, 0]
    Ixy = (window[2, 2] - window[2, 0] - window[0, 2] + window[0, 0]) / 4

    # Construct the Hessian matrix
    H = np.array([
        [Ixx, Ixy],
        [Ixy, Iyy]
    ])
    return H


# Iterate over each pixel in the 3x3 window within the 5x5 image
for i in range(1, 4):
    for j in range(1, 4):
        # Extract the 3x3 window around each pixel (i, j)
        window = image[i - 1:i + 2, j - 1:j + 2]
        hessian_matrix = compute_hessian_matrix(window)

        print(f"Hessian Matrix at pixel ({i},{j}) in the 3x3 window:")
        print(hessian_matrix)
        print()
