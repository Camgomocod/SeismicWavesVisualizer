from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QComboBox, QFrame, QMessageBox, 
                           QFileDialog, QApplication)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor, QImage, QPainter
import pyqtgraph as pg
import pyqtgraph.exporters  # Importación explícita del módulo de exportadores
import numpy as np
from scipy import signal
import pandas as pd
import io
from PIL import Image

class ValidationPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.setMaximumHeight(80)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title with status
        header_layout = QHBoxLayout()
        self.title_label = QLabel("Label Validation")
        self.title_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.status_label = QLabel()
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.status_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Details
        self.details_label = QLabel()
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)
        
        self.hide()  # Hidden by default
        
    def update_validation(self, validation_result):
        """Update the panel with validation results"""
        self.show()
        
        details = validation_result['details']
        has_p_arrival = details.get('has_p_arrival', False)
        
        if not has_p_arrival:
            # Case: No P arrival time data
            self.status_label.setText("ℹ No P arrival")
            self.status_label.setStyleSheet("color: blue;")
            self.details_label.setText(
                "No P arrival time data available for this signal. "
                f"Signal duration: {details['duration']:.2f}s"
            )
        elif validation_result['is_valid']:
            # Case: Valid P arrival time
            self.status_label.setText("✓ Valid")
            self.status_label.setStyleSheet("color: green;")
            self.details_label.setText(
                f"P arrival time ({details['relative_p_time']:.2f}s) "
                f"is within signal duration (0 to {details['duration']:.2f}s)"
            )
        else:
            # Case: Invalid P arrival time
            self.status_label.setText("⚠ Invalid")
            self.status_label.setStyleSheet("color: red;")
            self.details_label.setText(validation_result['error'])

class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(5)  # Reduce spacing between elements
        
        # Create top panel with metrics and controls in a more compact layout
        top_panel = QHBoxLayout()
        top_panel.setSpacing(10)  # Add some spacing between elements
        
        # Create info panel for signal metrics in a horizontal layout
        metrics_frame = QFrame()
        metrics_frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
        metrics_layout = QHBoxLayout(metrics_frame)
        metrics_layout.setContentsMargins(5, 2, 5, 2)  # Reduce margins
        
        # Create labels with fixed width for better alignment
        self.max_amp_label = QLabel("Max Amp: --")
        self.max_amp_label.setFixedWidth(150)
        self.snr_label = QLabel("SNR: --")
        self.snr_label.setFixedWidth(150)
        self.energy_label = QLabel("Energy Ratio: --")
        self.energy_label.setFixedWidth(150)
        
        metrics_layout.addWidget(self.max_amp_label)
        metrics_layout.addWidget(self.snr_label)
        metrics_layout.addWidget(self.energy_label)
        metrics_layout.addStretch()  # Add stretch to keep labels aligned to the left
        
        top_panel.addWidget(metrics_frame)
        
        # Add spectrogram controls
        spec_controls = QHBoxLayout()
        spec_controls.setSpacing(5)  # Reduce spacing
        
        self.spec_button = QPushButton("Show Spectrogram")
        self.spec_button.setFixedWidth(120)  # Fix width for consistency
        self.spec_button.clicked.connect(self.toggle_spectrogram)
        
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['viridis', 'plasma', 'magma', 'inferno', 'turbo', 'jet'])
        self.colormap_combo.setFixedWidth(100)  # Fix width for consistency
        self.colormap_combo.currentTextChanged.connect(self.update_colormap)
        self.colormap_combo.hide()  # Hide initially
        
        spec_controls.addWidget(self.spec_button)
        spec_controls.addWidget(self.colormap_combo)
        
        top_panel.addLayout(spec_controls)
        top_panel.addStretch()  # Add stretch to keep everything aligned to the left
        
        # Add export controls
        export_controls = QHBoxLayout()
        export_controls.setSpacing(5)
        
        self.export_button = QPushButton("Export Data")
        self.export_button.setToolTip("Export current data to CSV file")
        self.export_button.setFixedWidth(100)
        self.export_button.clicked.connect(self.export_data)
        
        self.save_plot_button = QPushButton("Save Plot")
        self.save_plot_button.setToolTip("Save current plot as image")
        self.save_plot_button.setFixedWidth(100)
        self.save_plot_button.clicked.connect(self.save_plot)
        
        export_controls.addWidget(self.export_button)
        export_controls.addWidget(self.save_plot_button)
        export_controls.addStretch()
        
        top_panel.addLayout(export_controls)
        
        # Add top panel to main layout
        self.layout.addLayout(top_panel)
        
        # Add validation panel
        self.validation_panel = ValidationPanel()
        self.layout.addWidget(self.validation_panel)
        
        # Create plot windows
        self.combined_plot = pg.PlotWidget(title="Seismic Signal")
        self.separate_plots = [
            pg.PlotWidget(title="Original Signal"),
            pg.PlotWidget(title="Filtered Signal")
        ]
        
        # Configure spectrogram plot
        self.spec_plot = pg.PlotWidget(title="Spectrogram")
        self.spec_plot.setLabel('left', 'Frequency (Hz)')
        self.spec_plot.setLabel('bottom', 'Time (s)')
        self.spec_img = pg.ImageItem()
        self.spec_plot.addItem(self.spec_img)
        self.spec_plot.hide()
        
        # Add combined plot initially
        self.layout.addWidget(self.combined_plot)
        
        # Configure plots
        for plot in [self.combined_plot] + self.separate_plots:
            plot.showGrid(x=True, y=True)
            plot.setLabel('left', 'Amplitude')
            plot.setLabel('bottom', 'Time (s)')
            plot.addLegend()
            
            # Enable mouse interaction
            plot.setMouseEnabled(x=True, y=True)
            
            # Add crosshair cursor
            vLine = pg.InfiniteLine(angle=90, movable=False)
            hLine = pg.InfiniteLine(angle=0, movable=False)
            plot.addItem(vLine, ignoreBounds=True)
            plot.addItem(hLine, ignoreBounds=True)
            
            def mouseMoved(evt):
                if plot.sceneBoundingRect().contains(evt):  # evt is already a QPointF
                    mousePoint = plot.getViewBox().mapSceneToView(evt)
                    vLine.setPos(mousePoint.x())
                    hLine.setPos(mousePoint.y())
            
            plot.scene().sigMouseMoved.connect(mouseMoved)
        
        # Hide separate plots initially
        for plot in self.separate_plots:
            plot.hide()
            self.layout.addWidget(plot)
        
        # Add spectrogram plot
        self.layout.addWidget(self.spec_plot)
        
        self.separate_view = False
        self.show_spectrogram = False
        self.current_colormap = 'plasma'
        self.current_file_id = None  # Añadimos esta variable para almacenar el ID
        
    def update_colormap(self, colormap_name):
        """Update the colormap of the spectrogram"""
        self.current_colormap = colormap_name
        if self.show_spectrogram and hasattr(self, '_last_plot_data'):
            self.update_spectrogram(
                self._last_plot_data['original_data'],
                self._last_plot_data['filtered_data']
            )
    
    def toggle_view(self):
        """Toggle between combined and separate views"""
        self.separate_view = not self.separate_view
        
        if self.separate_view:
            self.combined_plot.hide()
            for plot in self.separate_plots:
                plot.show()
        else:
            self.combined_plot.show()
            for plot in self.separate_plots:
                plot.hide()
                
        # Re-plot the data if we have it
        if hasattr(self, '_last_plot_data'):
            self.plot_data(**self._last_plot_data)
    
    def toggle_spectrogram(self):
        """Toggle spectrogram view"""
        self.show_spectrogram = not self.show_spectrogram
        self.spec_plot.setVisible(self.show_spectrogram)
        self.colormap_combo.setVisible(self.show_spectrogram)
        self.spec_button.setText("Hide Spectrogram" if self.show_spectrogram else "Show Spectrogram")
        
        # Update spectrogram if we have data
        if hasattr(self, '_last_plot_data') and self.show_spectrogram:
            self.update_spectrogram(
                self._last_plot_data['original_data'],
                self._last_plot_data['filtered_data']
            )
    
    def calculate_metrics(self, data, p_arrival_time=None, sampling_rate=100):
        """Calculate signal metrics"""
        max_amp = np.max(np.abs(data))
        
        if p_arrival_time is not None:
            # Convert p_arrival_time from seconds to samples
            p_sample = int(p_arrival_time * sampling_rate)
            
            # Calculate noise and signal windows
            noise_window = data[max(0, p_sample-500):p_sample]
            signal_window = data[p_sample:min(len(data), p_sample+500)]
            
            # Calculate SNR
            noise_power = np.mean(noise_window ** 2)
            signal_power = np.mean(signal_window ** 2)
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            
            # Calculate signal energy
            energy_before = np.sum(noise_window ** 2)
            energy_after = np.sum(signal_window ** 2)
            
            return max_amp, snr, energy_before, energy_after
        
        return max_amp, None, None, None
    
    def update_spectrogram(self, original_data, filtered_data=None):
        """Update spectrogram plot"""
        if not self.show_spectrogram:
            return
        
        # Calculate spectrogram with improved parameters
        fs = 100  # sampling rate
        nperseg = min(256, len(original_data)//4)  # Adjust segment size
        noverlap = nperseg // 2  # 50% overlap
        
        f, t, Sxx = signal.spectrogram(
            original_data, 
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window='hann'
        )
        
        # Use logarithmic scale with proper normalization
        Sxx = 10 * np.log10(np.maximum(Sxx, 1e-10))
        
        # Update image
        self.spec_img.setImage(Sxx, autoLevels=True)
        self.spec_img.scale(t[-1]/Sxx.shape[1], f[-1]/Sxx.shape[0])
        
        # Update or create colorbar
        if hasattr(self, 'colorbar'):
            self.spec_plot.removeItem(self.colorbar)
        
        # Create new colorbar with current colormap
        self.colorbar = pg.ColorBarItem(
            values=(Sxx.min(), Sxx.max()),
            colorMap=self.current_colormap,
            label='Power/Frequency (dB/Hz)',
            width=10
        )
        self.colorbar.setImageItem(self.spec_img)
        
        # Set proper axis ranges
        self.spec_plot.setXRange(0, t[-1])
        self.spec_plot.setYRange(0, f[-1])

    def plot_data(self, times, original_data, filtered_data=None, p_arrival_time=None, file_id=None):
        """Plot original and filtered seismic data"""
        # Store the last plot data for view toggling
        self._last_plot_data = {
            'times': times,
            'original_data': original_data,
            'filtered_data': filtered_data,
            'p_arrival_time': p_arrival_time,
            'file_id': file_id  # Almacenamos también el ID del archivo
        }
        
        # Calculate and update metrics
        max_amp, snr, energy_before, energy_after = self.calculate_metrics(
            original_data, p_arrival_time, sampling_rate=1/np.mean(np.diff(times))
        )
        
        self.max_amp_label.setText(f"Max Amplitude: {max_amp:.2f}")
        if snr is not None:
            self.snr_label.setText(f"SNR: {snr:.2f} dB")
            self.energy_label.setText(f"Energy Ratio: {energy_after/energy_before:.2f}")
        
        # Clear all plots
        self.combined_plot.clear()
        for plot in self.separate_plots:
            plot.clear()
        
        if self.separate_view and filtered_data is not None:
            # Plot in separate views
            self.separate_plots[0].plot(times, original_data, pen='b', name='Original')
            self.separate_plots[1].plot(times, filtered_data, pen='orange', name='Filtered')
            
            # Add P arrival markers if available
            if p_arrival_time is not None:
                for plot in self.separate_plots:
                    p_line = pg.InfiniteLine(pos=p_arrival_time, angle=90, pen='r', label='P arrival')
                    plot.addItem(p_line)
                    text = pg.TextItem("P arrival", color='r', anchor=(0.5, 1.0))
                    text.setPos(p_arrival_time, plot.getViewBox().viewRange()[1][1])
                    plot.addItem(text)
        else:
            # Plot in combined view
            self.combined_plot.plot(times, original_data, pen='b', name='Original')
            if filtered_data is not None:
                self.combined_plot.plot(times, filtered_data, pen='orange', name='Filtered')
            
            # Add P arrival marker if available
            if p_arrival_time is not None:
                p_line = pg.InfiniteLine(pos=p_arrival_time, angle=90, pen='r', label='P arrival')
                self.combined_plot.addItem(p_line)
                text = pg.TextItem("P arrival", color='r', anchor=(0.5, 1.0))
                text.setPos(p_arrival_time, self.combined_plot.getViewBox().viewRange()[1][1])
                self.combined_plot.addItem(text)
        
        # Update spectrogram if visible
        if self.show_spectrogram:
            self.update_spectrogram(original_data, filtered_data)
    
    def clear(self):
        """Clear all plots"""
        self.combined_plot.clear()
        for plot in self.separate_plots:
            plot.clear()
        if hasattr(self, 'spec_img'):
            self.spec_img.clear()
        
        # Reset labels
        self.max_amp_label.setText("Max Amplitude: --")
        self.snr_label.setText("SNR: --")
        self.energy_label.setText("Signal Energy: --")
    
    def set_validation_result(self, validation_result):
        """Update the validation panel with new results"""
        self.validation_panel.update_validation(validation_result)
    
    def export_data(self):
        """Export current data to CSV file"""
        if not hasattr(self, '_last_plot_data'):
            QMessageBox.warning(self, "Export Error", "No data available to export")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Data to CSV",
            "",
            "CSV Files (*.csv)"
        )
        
        if not file_path:
            return
            
        try:
            data = {
                'Time (s)': self._last_plot_data['times'],
                'Original': self._last_plot_data['original_data']
            }
            
            if self._last_plot_data['filtered_data'] is not None:
                data['Filtered'] = self._last_plot_data['filtered_data']
            
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            QMessageBox.information(self, "Success", "Data exported successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export data: {str(e)}")
    
    def save_plot(self):
        """Save current plot as image"""
        if not hasattr(self, '_last_plot_data'):
            QMessageBox.warning(self, "Save Error", "No plot available to save")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            "",
            "PNG Files (*.png)"
        )
        
        if not file_path:
            return
            
        try:
            # Ensure file has correct extension
            if not file_path.lower().endswith('.png'):
                file_path += '.png'
            
            # Get file ID for the title
            file_id = self._last_plot_data.get('file_id', 'Unknown')
            
            if self.separate_view:
                # Create a QWidget to hold both plots with proper styling
                temp_widget = QWidget()
                temp_widget.setStyleSheet("background-color: black;")
                layout = QHBoxLayout(temp_widget)
                layout.setSpacing(0)
                layout.setContentsMargins(10, 10, 10, 10)
                
                # Create new plot widgets with proper styling
                plot1 = pg.PlotWidget(title=f"Original Signal - File ID: {file_id}", background='k')
                plot2 = pg.PlotWidget(title=f"Filtered Signal - File ID: {file_id}", background='k')
                
                # Configure plots
                for plot in [plot1, plot2]:
                    plot.showGrid(x=True, y=True, alpha=0.3)
                    plot.setLabel('left', 'Amplitude', color='white')
                    plot.setLabel('bottom', 'Time (s)', color='white')
                    plot.getAxis('left').setPen(pg.mkPen(color='white'))
                    plot.getAxis('bottom').setPen(pg.mkPen(color='white'))
                    plot.getAxis('left').setTextPen(pg.mkPen(color='white'))
                    plot.getAxis('bottom').setTextPen(pg.mkPen(color='white'))
                    plot.setTitle(color='white', size='12pt')
                    plot.setBackground('k')
                
                # Add plots to layout
                layout.addWidget(plot1)
                layout.addWidget(plot2)
                
                # Plot data with improved styling
                plot1.plot(self._last_plot_data['times'], 
                         self._last_plot_data['original_data'],
                         pen=pg.mkPen(color='b', width=1.5),
                         name='Original')
                         
                if self._last_plot_data['filtered_data'] is not None:
                    plot2.plot(self._last_plot_data['times'],
                             self._last_plot_data['filtered_data'],
                             pen=pg.mkPen(color=(255, 165, 0), width=1.5),
                             name='Filtered')
                
                # Add P arrival markers if available
                if self._last_plot_data['p_arrival_time'] is not None:
                    p_time = self._last_plot_data['p_arrival_time']
                    for plot in [plot1, plot2]:
                        p_line = pg.InfiniteLine(pos=p_time, angle=90, 
                                               pen=pg.mkPen('r', width=2, style=Qt.DashLine))
                        plot.addItem(p_line)
                        text = pg.TextItem("P arrival", color='r', anchor=(0.5, 1.0))
                        text.setPos(p_time, plot.getViewBox().viewRange()[1][1])
                        plot.addItem(text)
                
                # Set size for high quality
                temp_widget.resize(1600, 600)
                
                # Set axis ranges to match current view
                y_range = self.separate_plots[0].getViewBox().viewRange()[1]
                plot1.setYRange(y_range[0], y_range[1])
                plot2.setYRange(y_range[0], y_range[1])
                
                # Show widget and wait for it to render
                temp_widget.show()
                QApplication.processEvents()
                
                # Create high resolution image
                pixmap = temp_widget.grab()
                pixmap.save(file_path, quality=100)
                
                # Clean up
                temp_widget.close()
                
            else:
                # Create a new plot widget for the combined view
                temp_widget = QWidget()
                temp_widget.setStyleSheet("background-color: black;")
                layout = QVBoxLayout(temp_widget)
                plot = pg.PlotWidget(title=f"Seismic Signal - File ID: {file_id}", background='k')
                
                # Configure plot
                plot.showGrid(x=True, y=True, alpha=0.3)
                plot.setLabel('left', 'Amplitude', color='white')
                plot.setLabel('bottom', 'Time (s)', color='white')
                plot.getAxis('left').setPen(pg.mkPen(color='white'))
                plot.getAxis('bottom').setPen(pg.mkPen(color='white'))
                plot.getAxis('left').setTextPen(pg.mkPen(color='white'))
                plot.getAxis('bottom').setTextPen(pg.mkPen(color='white'))
                plot.setTitle(color='white', size='12pt')
                plot.setBackground('k')
                
                # Add plot to layout
                layout.addWidget(plot)
                
                # Plot data with improved styling
                plot.plot(self._last_plot_data['times'], 
                        self._last_plot_data['original_data'],
                        pen=pg.mkPen(color='b', width=1.5),
                        name='Original')
                
                if self._last_plot_data['filtered_data'] is not None:
                    plot.plot(self._last_plot_data['times'],
                            self._last_plot_data['filtered_data'],
                            pen=pg.mkPen(color=(255, 165, 0), width=1.5),
                            name='Filtered')
                
                # Add P arrival marker if available
                if self._last_plot_data['p_arrival_time'] is not None:
                    p_line = pg.InfiniteLine(pos=self._last_plot_data['p_arrival_time'],
                                           angle=90,
                                           pen=pg.mkPen('r', width=2, style=Qt.DashLine))
                    plot.addItem(p_line)
                    text = pg.TextItem("P arrival", color='r', anchor=(0.5, 1.0))
                    text.setPos(self._last_plot_data['p_arrival_time'],
                              plot.getViewBox().viewRange()[1][1])
                    plot.addItem(text)
                
                # Set size and match current view range
                temp_widget.resize(800, 600)
                y_range = self.combined_plot.getViewBox().viewRange()[1]
                plot.setYRange(y_range[0], y_range[1])
                
                # Show and render
                temp_widget.show()
                QApplication.processEvents()
                
                # Create high resolution image
                pixmap = temp_widget.grab()
                pixmap.save(file_path, quality=100)
                
                # Clean up
                temp_widget.close()
            
            QMessageBox.information(self, "Success", f"Plot saved successfully to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save plot: {str(e)}")
            print(f"Detailed error: {str(e)}")