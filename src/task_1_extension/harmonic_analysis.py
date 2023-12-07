import numpy as np
import matplotlib.pyplot as plt


def plot_fft(fft_result, title):
    """
    Plot the FFT of a 2D array.
    :param fft_result: The result of np.fft.fft2()
    :param title: The title of the plot
    :return: None
    """

    fig, ax = plt.subplots()

    # Shift the zero-frequency component to the center for visualization
    fft_shifted = np.fft.fftshift(fft_result)

    # Calculate the magnitude (absolute value)
    magnitude = np.abs(fft_shifted)

    # Scale the magnitude to make it easier to see
    magnitude_scaled = np.log1p(magnitude)

    cax = ax.imshow(magnitude_scaled, cmap='gray_r', interpolation='none', extent=[0, magnitude_scaled.shape[1], 0, magnitude_scaled.shape[0]])

    ax.set_title(title)

    cbar = fig.colorbar(cax, ax=ax)

    plt.show()


def convolve(a, w, plot=False):
    """
    Perform the convolution of a and w.
    :param a: Object
    :param w: Workspace
    :param plot: Whether to plot the FFTs
    :return: The convolution of a and w
    """

    # Perform the FFTs
    w_fft = np.fft.fft2(w)

    a_flipped = np.flip(a, axis=1)
    a_flipped_fft = np.fft.fft2(a_flipped)

    # Perform the point-wise product
    point_wise_product = w_fft * a_flipped_fft

    # Perform the inverse FFT
    convolution = np.fft.ifft2(point_wise_product)

    # Take the real part of the result
    convolution = np.real(convolution)

    # Plot the FFT of the workspace
    if plot:
        plot_fft(w_fft, 'FFT of Workspace')
        plot_fft(a_flipped_fft, 'FFT of Flipped Object')
        plot_fft(point_wise_product, 'Point-wise Product')

    return convolution


def determine_optimal_offset(w, a):
    """
    (Deprecated) Determine the optimal offset for placing object a in workspace w.
    :param w: Workspace
    :param a: Object
    :return: The optimal offset
    """

    convolution_wa = convolve(w, a)

    # Trim the convolution to prevent object placement outside the workspace

    # Find the bottom-leftmost zero in the convolution
    bottom_leftmost_zero = np.argwhere(convolution_wa == 0)[0]

    # Determine the offset
    offset = np.array([a.shape[0] - bottom_leftmost_zero[0], bottom_leftmost_zero[1]])

    return offset