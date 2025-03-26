# Audio Feature Extraction Guide

This document outlines a set of standard techniques and Python code snippets to extract meaningful features from audio signals. These features are commonly used in speech processing tasks such as speaker recognition, gender detection, speech emotion recognition, and more.

## 1. Statistical Features

Compute basic descriptive statistics of the signal's frequency content:

```python
import numpy as np
import scipy.stats

freqs = np.fft.fftfreq(x.size)

def describe_freq(freqs):
    mean = np.mean(freqs)
    std = np.std(freqs)
    maxv = np.amax(freqs)
    minv = np.amin(freqs)
    median = np.median(freqs)
    skew = scipy.stats.skew(freqs)
    kurt = scipy.stats.kurtosis(freqs)
    q1 = np.quantile(freqs, 0.25)
    q3 = np.quantile(freqs, 0.75)
    mode = scipy.stats.mode(freqs)[0][0]
    iqr = scipy.stats.iqr(freqs)

    return [mean, std, maxv, minv, median, skew, kurt, q1, q3, mode, iqr]
```
These can be computed per window or over the whole signal.

---

## 2. Energy

Defined as the sum of squared magnitudes of the signal:

```python
def energy(x):
    return np.sum(x**2)
```

---

## 3. Root Mean Square Energy (RMSE)

```python
def rmse(x):
    return np.sqrt(np.mean(x**2))

# Or using Librosa:
import librosa
rmse = librosa.feature.rms(x)[0]
```

---

## 4. Zero-Crossing Rate (ZCR)

```python
zero_crossings = sum(librosa.zero_crossings(x, pad=False))
```

---

## 5. Tempo

Estimate tempo in Beats Per Minute (BPM):

```python
tempo = librosa.beat.tempo(x)[0]
```

---

## 6. Mel Frequency Cepstral Coefficients (MFCC)

Widely used in speech/audio processing. Extracted with:

```python
x, sr = librosa.load(filename)
mfcc = librosa.feature.mfcc(x)
```

To summarize the MFCCs over time:
```python
mfcc_summary = [
    np.mean(mfcc, axis=1),
    np.std(mfcc, axis=1),
    np.min(mfcc, axis=1),
    np.max(mfcc, axis=1),
    np.median(mfcc, axis=1),
]
```

---

## 7. MFCC Delta Coefficients

Delta coefficients capture the rate of change of MFCCs:

```python
delta_mfcc = librosa.feature.delta(mfcc)
```

---

## 8. Polyfeatures

Polynomial coefficients fitted to each frame:

```python
poly_features = librosa.feature.poly_features(x)  # Default: order=1
```

---

## 9. Tempogram

Measure tempo variations over time:

```python
hop_length = 512
oenv = librosa.onset.onset_strength(y=x, sr=sr, hop_length=hop_length)
tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
```

---

## 10. Spectral Features

Various descriptors of the spectral distribution:

```python
spec_centroid = librosa.feature.spectral_centroid(x)[0]
spectral_bandwidth = librosa.feature.spectral_bandwidth(x)[0]
spectral_contrast = librosa.feature.spectral_contrast(x)[0]
spectral_flatness = librosa.feature.spectral_flatness(x)[0]
spectral_rolloff = librosa.feature.spectral_rolloff(x)[0]
```

---

## 11. Fundamental Frequency

Useful for pitch-related features:

```python
f0, voiced_flag, voiced_probs = librosa.pyin(x, fmin=85, fmax=255)
```

---

## 12. Jitter Feature

Measures deviation of periodicity:

```python
def jitter(T):
    return np.mean(np.abs(np.diff(T)))
```

---

## Summary
These features can be computed either over entire audio files or using a sliding window approach for temporal resolution. They provide a rich representation of the signal and are fundamental in various audio and speech-related tasks.

Use libraries like `librosa`, `numpy`, and `scipy` to build an efficient feature extraction pipeline.

