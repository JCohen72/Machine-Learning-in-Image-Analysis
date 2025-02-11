# Machine Learning in Image Analysis

This repository contains several modular projects—each implemented as a Jupyter Notebook—that explore different machine learning and simulation techniques applied to image analysis. The implementations include Bayesian methods, numerical ODE integration, geodesic computations, frequency domain filtering, and total variation (TV) denoising. Each module is self-contained, includes detailed inline comments and docstrings, and is designed for both experimentation and real-world scalability.

## Table of Contents

- [Overview](#overview)
- [Modules](#modules)
  - [Bayesian Logistic Regression](#bayesian-logistic-regression)
  - [Euler Method Simulations](#euler-method-simulations)
  - [Geodesic Shooting Simulations](#geodesic-shooting-simulations)
  - [Image Frequency Smoothing](#image-frequency-smoothing)
  - [TV Model Gradient Descent Denoising](#tv-model-gradient-descent-denoising)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
- [License](#license)

## Overview

The projects in this repository are intended to illustrate a variety of techniques in machine learning and image analysis. Each module demonstrates a specific approach:
- **Probabilistic Classification:** Using Bayesian logistic regression to quantify uncertainty.
- **Numerical Simulation:** Employing Euler’s method to integrate differential equations.
- **Shape Analysis:** Simulating geodesic shooting to model deformations.
- **Frequency Domain Filtering:** Smoothing images by manipulating their Fourier transforms.
- **Image Denoising:** Applying gradient descent to minimize a total variation energy for robust denoising.

Each notebook is written with an emphasis on clean coding practices, including clear naming conventions, modular structure, and comprehensive documentation via inline comments and docstrings.

## Modules

### Bayesian Logistic Regression

**Contents:**  
- `Bayesian Logistic Regression/Bayesian Logistic Regression.ipynb`

**Description:**  
This notebook demonstrates how to perform logistic regression within a Bayesian framework. Key features include:
- **Model Specification:** Defining prior distributions and likelihood functions.
- **Inference:** Estimating posterior distributions using either analytical approximations or sampling techniques.
- **Visualization:** Plotting decision boundaries, uncertainty estimates, and probability distributions.

**Highlights:**  
- Clear modular code segments for data preprocessing, model definition, and result visualization.
- Docstrings for every function to explain parameters and return values.

---

### Euler Method Simulations

**Contents:**  
- `Euler Method Simulations/Euler Method Simulations.ipynb`

**Description:**  
This module showcases the use of Euler’s method for numerical integration of ordinary differential equations (ODEs). It includes:
- **ODE Definition:** Simple example ODEs with known analytical solutions.
- **Iteration & Error Analysis:** Step-by-step simulation with adjustable time-step parameters and comparison with analytical benchmarks.
- **Visualization:** Graphs illustrating the trajectory of the simulated system and error convergence.

**Highlights:**  
- Modular functions for the ODE solver.
- Detailed comments and docstrings explaining the numerical method and its limitations.

---

### Geodesic Shooting Simulations

**Contents:**  
- `Geodesic Shooting Simulations/Geodesic Shooting Simulations.ipynb`

**Description:**  
This notebook implements a simulation of geodesic shooting—a technique used in image registration and shape analysis. The code:
- **Formulates Geodesic Equations:** Sets up the differential equations governing geodesic paths on a manifold.
- **Numerical Integration:** Uses iterative integration (similar to Euler’s method) to compute the geodesic path.
- **Visualization:** Plots the trajectory of deformations to aid in understanding the underlying geometry.

**Highlights:**  
- Clear separation between the mathematical formulation and numerical solution.
- Inline explanations that bridge theory and implementation.

---

### Image Frequency Smoothing

**Contents:**  
- `Image Frequency Smoothing/Image Frequency Smoothing.ipynb`

**Description:**  
This notebook applies frequency domain filtering to smooth images. It demonstrates:
- **Fourier Transform:** Converting images into their frequency components using the Fast Fourier Transform (FFT).
- **Filtering:** Attenuating high-frequency noise components and reconstructing the image.
- **Visualization:** Side-by-side comparisons of the original and smoothed images, along with frequency spectrum plots.

**Highlights:**  
- Adjustable filter parameters for experimenting with different smoothing effects.
- Comprehensive visualization routines to highlight the impact of frequency filtering.

---

### TV Model Gradient Descent Denoising

**Contents:**  
- `TV Model Gradient Descent Denoising/TV Model Gradient Descent Denoising.ipynb`

**Description:**  
This module implements image denoising based on a Total Variation (TV) model, optimized via gradient descent. It features:
- **Energy Function Definition:** Combining a fidelity term with a regularization term (TV norm) to preserve edges.
- **Gradient Descent Optimization:** Iteratively updating image pixel values to minimize the energy function.
- **Result Comparison:** Visualization of the noisy input versus the denoised output, including convergence plots for the energy minimization.

**Highlights:**  
- Functions are documented with clear docstrings explaining their roles in the denoising process.
- Emphasis on parameter tuning (e.g., learning rate, iteration count) to balance noise removal and detail preservation.

---

## Installation & Requirements

Clone the repository:

```bash
git clone https://github.com/JCohen72/Machine-Learning-in-Image-Analysis.git
cd Machine-Learning-in-Image-Analysis
```

### Python Dependencies

The notebooks rely on the following libraries:
- **numpy**
- **matplotlib**
- **scikit-image**
- **opencv-python**
- **Pillow**

Install them via pip:

```bash
pip install numpy matplotlib scikit-image opencv-python Pillow
```

*Note: Some notebooks may have additional library dependencies. Check the first cell of each notebook for specific requirements.*

---

## Usage

1. **Launch Jupyter Notebook:**  
   Run the following command in your terminal to start Jupyter:
   ```bash
   jupyter notebook
   ```
2. **Navigate to the Module:**  
   Open any notebook (e.g., `Bayesian Logistic Regression.ipynb`) to explore the code and visualizations.
3. **Experiment:**  
   Modify parameters, test with different datasets, or adjust simulation settings as needed.

---

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
