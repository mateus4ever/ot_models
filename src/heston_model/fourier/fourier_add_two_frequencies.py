import numpy as np
import matplotlib.pyplot as plt

# Signal parameters
sampling_rate = 10000  # Sampling rate in Hz
duration = 1.0  # Duration in seconds
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Generate two sine waves
freq1 = 1000  # Frequency of the first sine wave in Hz
freq2 = 2000  # Frequency of the second sine wave in Hz
signal1 = np.sin(2 * np.pi * freq1 * t)
signal2 = np.sin(2 * np.pi * freq2 * t)

# Perform FFT on both signals
fft_signal1 = np.fft.fft(signal1)
fft_signal2 = np.fft.fft(signal2)

# Add the signals in the frequency domain
fft_combined = fft_signal1 + fft_signal2

# Inverse FFT to combine the signals back in the time domain
combined_signal = np.fft.ifft(fft_combined)

# Frequency axis
fft_freqs = np.fft.fftfreq(len(t), d=1/sampling_rate)

# Plot the results
plt.figure(figsize=(12, 8))

# Time-domain signals
plt.subplot(3, 1, 1)
plt.plot(t[:1000], signal1[:1000], label="Signal 1 (1000 Hz)")
plt.plot(t[:1000], signal2[:1000], label="Signal 2 (2000 Hz)")
plt.title("Original Signals in Time Domain")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Frequency-domain signals
plt.subplot(3, 1, 2)
plt.plot(fft_freqs[:len(fft_freqs)//2], np.abs(fft_signal1[:len(fft_signal1)//2]), label="FFT Signal 1")
plt.plot(fft_freqs[:len(fft_freqs)//2], np.abs(fft_signal2[:len(fft_signal2)//2]), label="FFT Signal 2")
plt.title("Signals in Frequency Domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Combined signal in time domain
plt.subplot(3, 1, 3)
plt.plot(t[:1000], combined_signal[:1000].real, label="Combined Signal")
plt.title("Combined Signal in Time Domain")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()