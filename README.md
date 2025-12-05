# FFTed - Interactive FFT & DCT Image Visualizer

A powerful desktop application for real-time frequency domain analysis of images using Fast Fourier Transform (FFT) and Discrete Cosine Transform (DCT) with interactive 3D visualizations.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyQt6](https://img.shields.io/badge/PyQt-6-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

### Dual Transform Analysis
- **FFT (Fast Fourier Transform)**: Analyze frequency components with optional FFT shifting
- **DCT (Discrete Cosine Transform)**: Side-by-side DCT visualization for comparison
- **Real-time Processing**: Background computation with automatic cancellation of outdated requests

### Interactive 3D Visualization
Choose from three visualization modes:
- **Unit Height**: All points at z=1, perfect for viewing the frequency distribution pattern
- **Magnitude Height**: Height represents magnitude, ideal for identifying dominant frequencies
- **Normalized Mode**: Normalized coordinates with magnitude-based height for balanced view

### Advanced Filtering System
- **Dynamic Filters**: Add multiple frequency or amplitude-based band-stop filters
- **Real-time Preview**: See filtered results instantly as you adjust parameters
- **Dual Filter Types**:
  - Frequency-domain filtering (by radial frequency)
  - Amplitude-domain filtering (by magnitude values)

### Power Spectral Density (PSD)
- 2D heatmap visualization of frequency power distribution
- Logarithmic or linear scale options
- Color-coded intensity mapping

### Image Reconstruction
- Real-time reconstruction from filtered frequency domain
- Export reconstructed images at original resolution
- Automatic scaling and normalization

## 🚀 Installation

### Prerequisites
```bash
python >= 3.8
```

### Install Dependencies
```bash
pip install numpy opencv-python scipy PyQt6 matplotlib
```

Or using requirements.txt:
```bash
pip install -r requirements.txt
```

### Requirements.txt
```
numpy>=1.20.0
opencv-python>=4.5.0
scipy>=1.7.0
PyQt6>=6.0.0
matplotlib>=3.4.0
```

## 📖 Usage

### Running the Application
```bash
python main.py
```

### Quick Start Guide

1. **Load an Image**
   - Click "Load Image" button
   - Select any grayscale or color image (auto-converted to grayscale)
   - Image will be displayed in the bottom-left panel

2. **Compute Transforms**
   - Click "Compute FFT" to analyze frequency domain
   - Click "Compute DCT" for discrete cosine transform
   - Both can run simultaneously for comparison

3. **Explore Visualization Modes**
   - **Unit Height**: Best for seeing overall frequency distribution
   - **Magnitude Height**: Emphasizes dominant frequency components
   - **Normalized + Magnitude**: Balanced view with normalized coordinates

4. **Apply Filters**
   - Click "+ Add filter" to create a new filter
   - Choose filter type: `freq` (frequency) or `amp` (amplitude)
   - Set low and high range values
   - Enable/disable filters with checkboxes
   - Watch real-time reconstruction updates

5. **Export Results**
   - Click "Export Image" to save the reconstructed image
   - Output is automatically scaled to original image dimensions

### Interface Layout

```
┌─────────────────────────────────────┬──────────────┐
│  3D FFT Visualization (Left)        │   Control    │
│  3D DCT Visualization (Right)       │   Panel      │
├─────────────────────────────────────┤              │
│  Power Spectral Density (PSD)       │   • Load     │
├─────────────────────────────────────┤   • Compute  │
│  Original  │  FFT Recon │ DCT Recon │   • Options  │
│   Image    │   Image    │  Image    │   • Filters  │
└────────────┴────────────┴───────────┴──────────────┘
```

## 🎯 Use Cases

### Education
- **Signal Processing Courses**: Visualize FFT/DCT concepts in 2D
- **Image Compression**: Understand frequency domain compression (JPEG uses DCT)
- **Filter Design**: Experiment with frequency filtering effects

### Research & Analysis
- **Frequency Analysis**: Identify periodic patterns in images
- **Noise Reduction**: Design and test frequency-based noise filters
- **Image Enhancement**: Remove specific frequency bands (moiré patterns, scanning artifacts)

### Creative Applications
- **Artistic Effects**: Create unique filtered image effects
- **Pattern Recognition**: Analyze repeating patterns in textures
- **Quality Assessment**: Detect compression artifacts

## 🔧 Technical Details

### Architecture Highlights

**Asynchronous Processing**
- Background worker thread handles all heavy computations
- Latest-only request queue prevents UI blocking
- Automatic cancellation of stale requests when parameters change

**Performance Optimizations**
- Cached PSD computations to avoid redundant calculations
- Percentile-based axis limits for efficient rendering
- Smart sampling for 3D visualization (1/50th of resolution)

**Memory Management**
- Proper cleanup on application exit
- Reference nulling for large arrays
- Matplotlib figure cleanup to prevent memory leaks

### Key Algorithms

1. **2D FFT**: `numpy.fft.fft2()` with optional `fftshift` for centered frequency display
2. **2D DCT**: `scipy.fft.dct()` with orthonormal normalization
3. **Filtering**: Radial frequency masking or magnitude-based masking
4. **Reconstruction**: Inverse transforms with proper normalization

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- [ ] Add wavelet transform support
- [ ] Add phase visualization
- [ ] Interactive filter design with visual frequency selection
- [ ] Export capability for 3D plots

### Development Setup
```bash
git clone https://github.com/yourusername/ffted.git
cd ffted
pip install -r requirements.txt
python main.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) for the GUI framework
- [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/) for mathematical operations
- [Matplotlib](https://matplotlib.org/) for visualization
- [OpenCV](https://opencv.org/) for image I/O

## 📧 Contact

Have questions or suggestions? Feel free to open an issue or reach out!

---

**Note**: This tool is designed for educational and research purposes. For production image processing, consider specialized libraries like scikit-image or dedicated FFT tools.
