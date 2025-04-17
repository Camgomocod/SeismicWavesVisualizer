# Seismic Wave Visualizer

This project is a PyQt5 application designed to visualize seismic wave data. It provides a comprehensive interface to display, analyze, and process seismic signals, with robust filtering capabilities and data validation features.

## Features

### Signal Visualization and Analysis
- Interactive visualization of original and filtered seismic data
- Dual view modes: combined and separate views for better signal comparison
- Real-time display of signal metrics (Maximum Amplitude, SNR, Energy Ratio)
- Cross-hair cursor for precise measurements
- Dynamic zoom and pan capabilities
- Spectrogram visualization with customizable colormaps

### Signal Processing
- Butterworth bandpass filter with adjustable parameters:
  - Customizable low and high cutoff frequencies
  - Adjustable filter order
  - Real-time filter preview
- Save and load filter configurations for consistency across sessions

### Data Management
- Support for MSEED file format
- Automatic P-wave arrival time detection and visualization
- Validation of P-wave arrival times with detailed feedback
- Search and filter capabilities for seismic data files
- Keyboard shortcuts for efficient navigation

### Export and Documentation
- Export filtered and original data to CSV format
- Save plots as high-quality PNG images
- Automatic inclusion of file IDs and metadata in exports
- Documentation of signal parameters and processing steps

### User Interface
- Clean and intuitive PyQt5-based interface
- Dark theme optimized for signal visualization
- Status bar with informative feedback
- Comprehensive validation window for data quality control
- Tooltips and help text for all controls

## Installation

1. Clone the repository:
   ```
   git clone <git@github.com:Camgomocod/SeismicWavesVisualizer.git>
   cd seismic-wave-visualizer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt 
   ```

## Usage

To run the application, execute the following command:
```
python src/main.py
```

### Basic Controls
- Use Left/Right arrow keys to navigate between files
- Ctrl+F to apply filter
- Ctrl+V to toggle between views
- Click and drag to zoom, right-click to reset view

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.