# DeafVision
 CNN-based urban sound classification for AR glasses assistive technology. Part of the Deaf Vision research project (2020-2022).
The core idea: train a CNN to classify urban sounds from audio, then use that classification to provide real-time visual feedback through AR glasses — so deaf individuals can "see" the sounds around them.

## What It Does

Takes raw audio input, converts it into a mel spectrogram image, and classifies it into one of 10 urban sound categories using a convolutional neural network trained on the UrbanSound8K dataset.

## Sound Classes

| ID | Sound |
|---|---|
| 0 | Air Conditioner |
| 1 | Car Horn |
| 2 | Children Playing |
| 3 | Dog Bark |
| 4 | Drilling |
| 5 | Engine Idling |
| 6 | Gunshot |
| 7 | Jackhammer |
| 8 | Siren |
| 9 | Street Music |

## Dataset

**UrbanSound8K** — 8,732 labeled audio clips of urban sounds, each under 4 seconds, across 10 classes organized into 10 pre-defined folds for cross-validation.

## Model Architecture

Custom CNN built with TensorFlow/Keras:

```
Input (128 x 128 x 1)
Conv2D (32 filters, 3x3, stride 2, ReLU)
BatchNormalization + MaxPooling
Conv2D (128 filters, 3x3, ReLU)
BatchNormalization + MaxPooling
Flatten
Dense (256, ReLU) + BatchNormalization + Dropout (0.5)
Dense (10, Softmax)
```

Compiled with categorical crossentropy loss and RMSprop optimizer.

## Audio Processing Pipeline

Each audio file goes through the following before hitting the model:

1. Load audio at fixed duration (2.95 seconds)
2. Compute mel spectrogram using librosa (128 mel bands, hop length 512)
3. Convert to decibels using log scale (power to dB)
4. Pad or trim to fixed length of 128 frames
5. Reshape to (128, 128, 1) for CNN input

## Training

- 100 epochs with k-fold cross validation
- Data augmentation: feature-wise normalization, width shift, constant fill at -80.0 dB
- Evaluation includes accuracy and loss curves per fold
- Confusion matrix with seaborn heatmap for per-class analysis

## Project Structure

```
DeafVision/
├── downloading_and_processing_data.py    Audio loading and mel spectrogram extraction
├── model_training.py                     CNN architecture, training loop, evaluation
└── README.md
```

## Tech Stack

- Python
- TensorFlow and Keras for the CNN model
- librosa for audio processing and mel spectrogram computation
- NumPy and pandas for data handling
- scikit-learn for cross-validation and confusion matrix
- seaborn and matplotlib for visualization

## How to Run

```bash
pip install tensorflow librosa numpy pandas scikit-learn seaborn matplotlib tqdm
```

Run the data processing step first:
```bash
python downloading_and_processing_data.py
```

Then train the model:
```bash
python model_training.py
```

## Research Context

Deaf Vision was a two-year independent research project started in 2020. The goal was to build an assistive system where deaf individuals could receive real-time visual alerts about surrounding sounds through AR glasses. This repository contains the sound classification engine that powers that system.

The project started with image-based sign language recognition (see DeafTalk-2021) and evolved into this audio classification approach for broader environmental awareness.

Presented at the First World Cup for Innovation and Scientific Research, 2022. Competed in ISEF (International Science and Engineering Fair) nationally in Palestine, placing 1st and 4th across two competition cycles.
