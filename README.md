# Machine-Learning-in-Image-Analysis
---
# Image Processing with Frequency Smoothing and Gradient Descent Denoising

This repository contains Python implementations for image processing techniques focusing on frequency domain smoothing using Fourier transforms and image denoising using gradient descent optimization. The code is modular, implements good coding practices, and provides visualizations of the processed images.

## Table of Contents
- [Introduction](#introduction)
- [Files](#files)
- [Requirements](#requirements)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [License](#license)

## Introduction

This repository includes two Jupyter notebooks:

1. **Image Frequency Smoothing**: Applies Fourier transform-based techniques for image smoothing by filtering high-frequency components in the frequency domain.
2. **Gradient Descent Image Denoising**: Implements gradient descent to reduce noise in images by minimizing an energy function based on pixel gradients.

Both notebooks include visualizations that demonstrate the effect of the applied techniques on images.

## Files

### 1. `image_frequency_smoothing.ipynb`

**Description**:
This notebook focuses on smoothing images by transforming them into the frequency domain using the Fourier transform. It demonstrates the effect of removing or attenuating high-frequency components to reduce noise.

**Features**:
- Loads and displays an image.
- Computes the Fourier transform of the image to move it into the frequency domain.
- Applies various filters to smooth the image by attenuating specific frequency components.
- Visualizes the original and smoothed images along with their frequency representations.

**Code Overview**:
- **Image Loading**: Uses libraries like OpenCV and PIL to read images.
- **Fourier Transform**: Uses `numpy` to compute the 2D Fourier transform of images.
- **Filtering**: Applies frequency domain filtering to remove noise.
- **Plotting**: Visualizes the original and processed images with `matplotlib`.

### 2. `gradient_descent_image_denoising.ipynb`

**Description**:
This notebook demonstrates image denoising using a gradient descent algorithm. It minimizes an energy function based on the image’s pixel values and their gradients, which helps smooth out noise.

**Features**:
- Defines a custom function to compute the gradient of the image.
- Implements gradient descent to iteratively update pixel values and reduce noise.
- Compares the original noisy image to the denoised version after optimization.
  
**Code Overview**:
- **Gradient Computation**: Calculates image gradients to evaluate the smoothness of pixel intensities.
- **Gradient Descent Optimization**: Uses a step-by-step optimization approach to reduce noise.
- **Image Visualization**: Displays the noisy image, gradient map, and denoised image using `matplotlib`.

## Requirements

To run the notebooks, you need the following Python libraries:

- `numpy`
- `opencv-python`
- `matplotlib`
- `Pillow`
- `scikit-image`

You can install the required dependencies using pip:

```bash
pip install numpy opencv-python matplotlib Pillow scikit-image
```

## Usage

Clone the repository:

```bash
git clone https://github.com/yourusername/image-processing-smoothing-denoising.git
cd image-processing-smoothing-denoising
```

### Running the Notebooks

To run the notebooks, simply open them in a Jupyter environment:

1. **Image Frequency Smoothing**:
   - Open `image_frequency_smoothing.ipynb`.
   - Execute the cells to load an image, apply frequency smoothing, and visualize the results.

2. **Gradient Descent Image Denoising**:
   - Open `gradient_descent_image_denoising.ipynb`.
   - Execute the cells to load a noisy image, apply gradient descent denoising, and compare the results.

## Visualizations

### Image Frequency Smoothing:
- **Original vs. Smoothed Image**: Visual comparison of the original image and the result after applying frequency domain filtering.
- **Frequency Domain Visualization**: Shows the magnitude spectrum of the image before and after filtering.

### Gradient Descent Image Denoising:
- **Noisy vs. Denoised Image**: Visual comparison of the noisy input image and the output after gradient descent denoising.
- **Gradient Map**: Displays the computed gradients of the image to demonstrate the smoothing process.

## License

This repository is licensed under the MIT License. See the LICENSE file for more details.

---
