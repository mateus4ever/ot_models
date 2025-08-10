import numpy as np
import matplotlib.pyplot as plt

# Time domain: Create a signal (sine wave with noise)
t = np.linspace(0, 1, 1000, endpoint=False)  # Time from 0 to 1 second
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)  # Two sine waves
signal += np.random.normal(0, 0.2, signal.shape)  # Add noise

# Fourier Transform
freq = np.fft.fftfreq(len(t), d=t[1] - t[0])  # Frequencies
fft_signal = np.fft.fft(signal)  # FFT

# Plot
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("Time Domain Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(freq[:len(freq)//2], np.abs(fft_signal[:len(freq)//2]))  # Plot positive frequencies
plt.title("Frequency Domain (FFT)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()