import numpy as np
from skimage import data
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt


# Load an example image
image = data.camera()

# Compute Fourier Transform
fft_image = fft2(image)
fft_image_shifted = fftshift(fft_image)  # Center the zero-frequency component

# Plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Frequency Domain (Log Scale)")
plt.imshow(np.log1p(np.abs(fft_image_shifted)), cmap="gray")
plt.colorbar()
plt.tight_layout()
plt.show()