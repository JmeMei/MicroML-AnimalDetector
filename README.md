# TinyML Animal Sound Classifier

[![Platform: Arduino Nano 33 BLE](https://img.shields.io/badge/Platform-Arduino%20Nano%2033%20BLE-blue.svg)](https://store.arduino.cc/products/arduino-nano-33-ble-sense)
[![Framework: TensorFlow Lite](https://img.shields.io/badge/Framework-TensorFlow%20Lite%20Micro-orange.svg)](https://www.tensorflow.org/lite/microcontrollers)
[![Accuracy: 95.74%](https://img.shields.io/badge/Accuracy-95.74%25-green.svg)](#results-and-evaluation)

An end-to-end, ultra-low-power **TinyML Acoustic Classifier** deployed on an **Arduino Nano 33 BLE Sense**. This system identifies animal vocalizations across three distinct classes (**Dog, Cow, and Frog**) in real-time. By pairing specialized acoustic feature extraction with full 8-bit integer quantization, this project demonstrates a highly efficient, privacy-friendly, and cost-effective solution for smart livestock management and environmental monitoring.

---

## 🎥 Project Demonstrations
<p align="center">
  <video src="Docs/Assets/frog_detection.mp4" width="45%" controls muted></video>
  <video src="Docs/Assets/dog_detection.mp4" width="45%" controls muted></video>
</p>
---

## 📖 Table of Contents
* [System Architecture & Use Cases](#-system-architecture--use-cases)
* [Pipeline Overview](#%EF%B8%8F-pipeline-overview)
* [Model Architecture](#-model-architecture)
* [Optimization & Quantization](#-optimization--quantization)
* [Results and Evaluation](#-results-and-evaluation)
* [Deployment & Edge Logic](#-deployment--edge-logic)
* [Getting Started](#-getting-started)

---

## 🗺️ System Architecture & Use Cases

This project addresses practical needs in precision agriculture by processing audio directly on edge hardware, enabling real-time monitoring without relying on cloud infrastructure.

* **🐕 Dog Monitor (Security Trigger):** Detects barking to alert farmers to the presence of stray dogs or predators near livestock enclosures, especially during nocturnal hours.
* **🐄 Cow Monitor (Herd Welfare):** Detects cow vocalizations to monitor grazing patterns and distress signals. A significant increase in mooing frequency can indicate issues like calving difficulty or herd separation.
* **🐸 Frog Monitor (Eco-Biomarker):** Monitors frog vocalizations to assess water quality on the farm. Since frogs are highly sensitive to pollution, a decline in frog sounds can signal water contamination from agricultural runoff.

---

## ⚙️ Pipeline Overview

### 1. Audio Preprocessing & Noise Mitigation
Raw `.wav` source data is drawn from the Kaggle *Animal Sounds Classification (ASC24)* dataset. To isolate distinct acoustic signatures from background noise, audio is processed via `librosa` using two key filtering adjustments:
* **Silence Trimming:** Strips uninformative leading and trailing segments via `librosa.effects.trim(y, top_db=20)`.
* **Energy Filtering:** Computes a Root Mean Square (RMS) amplitude threshold to discard low-energy background snippets, ensuring the model trains only on active vocalizations.

### 2. Feature Extraction
Audio clips are sampled at **16 kHz** and peak-normalized within a range of `[-1, 1]`. They are chunked into uniform **3.2-second snippets** using a 512-sample frame size and a 256-sample hop length:

$$\text{Target Length} = 512 \times (200 - 1) \times 256 = 51,456 \text{ samples} \quad (\approx 3.216 \text{ seconds})$$

This outputs a 2D Mel-Frequency Cepstral Coefficient (**MFCC**) grid of shape `(13, 200)`. An extra channel parameter is appended to format the feature maps identically to a grayscale image, producing a final shape of **`(13, 200, 1)`**.

---

## 🧠 Model Architecture

The deep learning classifier is a lightweight Convolutional Neural Network (CNN) engineered to minimize parameters and memory footprint for edge microcontrollers like the Arduino Nano 33 BLE Sense. 

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

input_shape = (13, 200, 1)

model = keras.Sequential([
    layers.Input(shape=input_shape),
    
    layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    # Depthwise Convolution for parameter reduction
    layers.DepthwiseConv2D((3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    
    layers.GlobalAveragePooling2D(),
    
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(3, activation="softmax") # 3 target classes
])