from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QComboBox, QPushButton, QLabel,
                             QSpinBox, QDoubleSpinBox, QFileDialog, QLineEdit, QShortcut)
from PyQt5.QtCore import Qt, QStandardPaths
from PyQt5.QtGui import QKeySequence
import os

# Configure fallback for QStandardPaths
try:
    runtime_dir = QStandardPaths.writableLocation(QStandardPaths.RuntimeLocation)
    if not os.path.exists(runtime_dir):
        os.makedirs(runtime_dir, exist_ok=True)
except Exception:
    # Fallback to temp directory if runtime dir is not accessible
    os.environ['XDG_RUNTIME_DIR'] = '/tmp'

from src.ui.plot_widget import PlotWidget
from src.data.data_loader import DataLoader
from src.filters.butterworth import bandpass_filter
from src.ui.validation_window import ValidationWindow
import json
import pandas as pd

class MainWindow(QMainWindow):
    def __init__(self, csv_path=None):
        super().__init__()
        self.setWindowTitle("Seismic Wave Visualizer")
        self.resize(1200, 800)
        
        # Initialize data loader
        self.data_loader = DataLoader(csv_path)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create CSV file controls
        csv_controls = QHBoxLayout()
        self.csv_label = QLabel("P-wave Times CSV:")
        self.csv_path_label = QLabel(self.data_loader.csv_path or "No file selected")
        self.csv_path_label.setStyleSheet("color: gray;")
        self.select_csv_button = QPushButton("Select CSV")
        self.select_csv_button.setToolTip("Select CSV file with P-wave arrival times")
        self.select_csv_button.clicked.connect(self.select_csv_file)
        
        csv_controls.addWidget(self.csv_label)
        csv_controls.addWidget(self.csv_path_label)
        csv_controls.addWidget(self.select_csv_button)
        csv_controls.addStretch()
        
        layout.addLayout(csv_controls)
        
        # Create data directory controls with tooltips
        data_controls = QHBoxLayout()
        self.data_dir_label = QLabel("Data Directory:")
        self.data_dir_path = QLabel(self.data_loader.data_dir)
        self.select_dir_button = QPushButton("Select Directory")
        self.select_dir_button.setToolTip("Select the directory containing MSEED files")
        self.select_dir_button.clicked.connect(self.select_data_directory)
        
        data_controls.addWidget(self.data_dir_label)
        data_controls.addWidget(self.data_dir_path)
        data_controls.addWidget(self.select_dir_button)
        data_controls.addStretch()
        
        layout.addLayout(data_controls)
        
        # Create controls with tooltips
        controls_layout = QHBoxLayout()
        
        # File selection with tooltip
        self.file_combo = QComboBox()
        self.file_combo.setToolTip("Select a seismic data file to analyze")
        self.file_combo.addItems([str(x) for x in self.data_loader.get_file_list()])
        self.file_combo.currentTextChanged.connect(self.load_file)
        controls_layout.addWidget(QLabel("File:"))
        controls_layout.addWidget(self.file_combo)
        
        # Filter controls with tooltips
        controls_layout.addWidget(QLabel("Low Cut (Hz):"))
        self.low_cut = QDoubleSpinBox()
        self.low_cut.setToolTip("Lower frequency cutoff for the bandpass filter")
        self.low_cut.setRange(0.1, 10.0)
        self.low_cut.setValue(1.0)
        self.low_cut.valueChanged.connect(self.validate_filter_ranges)
        controls_layout.addWidget(self.low_cut)
        
        controls_layout.addWidget(QLabel("High Cut (Hz):"))
        self.high_cut = QDoubleSpinBox()
        self.high_cut.setToolTip("Upper frequency cutoff for the bandpass filter")
        self.high_cut.setRange(5.0, 50.0)
        self.high_cut.setValue(20.0)
        self.high_cut.valueChanged.connect(self.validate_filter_ranges)
        controls_layout.addWidget(self.high_cut)
        
        controls_layout.addWidget(QLabel("Filter Order:"))
        self.filter_order = QSpinBox()
        self.filter_order.setToolTip("Order of the Butterworth filter (higher = sharper cutoff)")
        self.filter_order.setRange(2, 8)
        self.filter_order.setValue(4)
        self.filter_order.valueChanged.connect(self.on_filter_changed)
        controls_layout.addWidget(self.filter_order)
        
        # Apply filter button with tooltip
        self.apply_button = QPushButton("Apply Filter")
        self.apply_button.setToolTip("Apply the current filter settings to the signal")
        self.apply_button.clicked.connect(self.apply_filter)
        controls_layout.addWidget(self.apply_button)
        
        # Toggle view button with tooltip
        self.toggle_view_button = QPushButton("Toggle View")
        self.toggle_view_button.setToolTip("Switch between combined and separate views")
        self.toggle_view_button.clicked.connect(self.toggle_view)
        controls_layout.addWidget(self.toggle_view_button)
        
        # Validation window button with tooltip
        self.validation_button = QPushButton("Open Validation")
        self.validation_button.setToolTip("Open window to validate P-wave arrival times")
        self.validation_button.clicked.connect(self.show_validation_window)
        controls_layout.addWidget(self.validation_button)
        
        # Add save/load filter config buttons
        filter_config_layout = QHBoxLayout()
        
        self.save_config_button = QPushButton("Save Filter Config")
        self.save_config_button.setToolTip("Save current filter settings")
        self.save_config_button.clicked.connect(self.save_filter_config)
        
        self.load_config_button = QPushButton("Load Filter Config")
        self.load_config_button.setToolTip("Load saved filter settings")
        self.load_config_button.clicked.connect(self.load_filter_config)
        
        filter_config_layout.addWidget(self.save_config_button)
        filter_config_layout.addWidget(self.load_config_button)
        filter_config_layout.addStretch()
        
        controls_layout.addLayout(filter_config_layout)
        
        # Add controls to main layout
        layout.addLayout(controls_layout)
        
        # Search layout with tooltip
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        self.search_input = QLineEdit()
        self.search_input.setToolTip("Enter file number to filter the file list")
        self.search_input.setPlaceholderText("Enter file number to search...")
        self.search_input.textChanged.connect(self.filter_files)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)
        
        # Create status bar for error messages
        self.statusBar().showMessage("Ready")
        
        # Create plot widget
        self.plot_widget = PlotWidget()
        layout.addWidget(self.plot_widget)
        
        # Store current data and filter state
        self.current_data = None
        self.current_file_id = None
        self.filter_active = False
        
        # Store validation window reference
        self.validation_window = None
        
        # Enable keyboard focus and shortcuts
        self.setFocusPolicy(Qt.StrongFocus)
        self.setup_shortcuts()
        
        # Add config file handling with better path management
        self.config_dir = os.path.join(os.path.expanduser("~"), ".seismic_visualizer")
        self.config_file = os.path.join(self.config_dir, "filter_config.json")
        
        # Create config directory if it doesn't exist
        os.makedirs(self.config_dir, exist_ok=True)
        
        self.load_last_config()
        
        # Load initial file
        if self.file_combo.count() > 0:
            self.load_file(self.file_combo.currentText())
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Navigation shortcuts
        self.shortcut_prev = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_prev.activated.connect(self.previous_file)
        
        self.shortcut_next = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_next.activated.connect(self.next_file)
        
        # Filter shortcuts
        self.shortcut_filter = QShortcut(QKeySequence("Ctrl+F"), self)
        self.shortcut_filter.activated.connect(self.apply_filter)
        
        self.shortcut_view = QShortcut(QKeySequence("Ctrl+V"), self)
        self.shortcut_view.activated.connect(self.toggle_view)
    
    def keyPressEvent(self, event):
        """Handle keyboard events for file navigation"""
        if event.key() == Qt.Key_Left:
            self.previous_file()
        elif event.key() == Qt.Key_Right:
            self.next_file()
        else:
            super().keyPressEvent(event)
    
    def previous_file(self):
        """Load the previous file in the list"""
        current_index = self.file_combo.currentIndex()
        if current_index > 0:
            self.file_combo.setCurrentIndex(current_index - 1)
    
    def next_file(self):
        """Load the next file in the list"""
        current_index = self.file_combo.currentIndex()
        if current_index < self.file_combo.count() - 1:
            self.file_combo.setCurrentIndex(current_index + 1)
    
    def validate_filter_ranges(self):
        """Ensure filter ranges are valid"""
        low = self.low_cut.value()
        high = self.high_cut.value()
        
        # Ensure low cut is always less than high cut
        if low >= high:
            if self.sender() == self.low_cut:
                self.low_cut.setValue(high - 0.1)
            else:
                self.high_cut.setValue(low + 0.1)
        
        if self.filter_active:
            self.apply_filter()
    
    def on_filter_changed(self):
        """Called when any filter parameter changes"""
        if self.filter_active:
            self.apply_filter()
    
    def load_file(self, file_id):
        """Load selected MSEED file and display it"""
        try:
            file_id = int(file_id)
            self.current_file_id = file_id
            
            # Get file path from data loader
            file_path = self.data_loader.get_mseed_path(file_id)
            
            if not os.path.exists(file_path):
                self.statusBar().showMessage(f"File not found: {file_path}", 5000)
                return
            
            try:
                # Load data
                self.current_data = self.data_loader.load_mseed(file_path)
                
                # Get P arrival time (may be None if not available)
                p_arrival = self.data_loader.get_p_arrival(file_id)
                p_time = None
                if p_arrival:
                    p_time = p_arrival - self.current_data['start_time']
                
                # Validate labels
                validation_result = self.data_loader.validate_labels(file_id, self.current_data)
                self.plot_widget.set_validation_result(validation_result)
                
                # Display data
                if self.filter_active:
                    filtered_data = bandpass_filter(
                        self.current_data['data_normalized'],
                        self.current_data['sampling_rate'],
                        lowcut=self.low_cut.value(),
                        highcut=self.high_cut.value(),
                        order=self.filter_order.value()
                    )
                else:
                    filtered_data = None
                
                self.plot_widget.plot_data(
                    self.current_data['times'],
                    self.current_data['data_normalized'],
                    filtered_data,
                    p_arrival_time=p_time,
                    file_id=file_id
                )
                
                self.statusBar().showMessage(f"Loaded file {file_id}", 3000)
                
            except Exception as e:
                self.statusBar().showMessage(f"Error processing file: {str(e)}", 5000)
                self.current_data = None
            
        except ValueError:
            self.statusBar().showMessage("Invalid file ID format", 5000)
        except Exception as e:
            self.statusBar().showMessage(f"Error: {str(e)}", 5000)
    
    def apply_filter(self):
        """Apply Butterworth filter with current settings"""
        if self.current_data is None:
            self.statusBar().showMessage("No data loaded to filter", 3000)
            return
            
        try:
            # Get filter parameters
            low = self.low_cut.value()
            high = self.high_cut.value()
            order = self.filter_order.value()
            
            # Apply filter
            filtered_data = bandpass_filter(
                self.current_data['data_normalized'],
                self.current_data['sampling_rate'],
                lowcut=low,
                highcut=high,
                order=order
            )
            
            # Get P arrival time
            p_arrival = self.data_loader.get_p_arrival(self.current_file_id)
            p_time = None
            if p_arrival:
                p_time = p_arrival - self.current_data['start_time']
            
            # Update plot
            self.plot_widget.plot_data(
                self.current_data['times'],
                self.current_data['data_normalized'],
                filtered_data,
                p_arrival_time=p_time,
                file_id=self.current_file_id
            )
            
            # Mark filter as active
            self.filter_active = True
            self.statusBar().showMessage("Filter applied successfully", 3000)
            
        except Exception as e:
            self.statusBar().showMessage(f"Error applying filter: {str(e)}", 5000)
            self.filter_active = False
    
    def toggle_view(self):
        """Toggle between combined and separate views"""
        self.plot_widget.toggle_view()
    
    def select_data_directory(self):
        """Open directory selection dialog"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Data Directory",
            self.data_loader.data_dir,
            QFileDialog.ShowDirsOnly
        )
        
        if directory:
            self.data_loader.set_data_directory(directory)
            self.data_dir_path.setText(directory)
            
            # Update file list in combo box
            self.file_combo.clear()
            file_list = self.data_loader.get_file_list()
            self.file_combo.addItems([str(x) for x in file_list])
            
            # Load first file if available
            if self.file_combo.count() > 0:
                self.load_file(self.file_combo.currentText())

    def show_validation_window(self):
        """Show the validation window"""
        if self.validation_window is None:
            self.validation_window = ValidationWindow(self.csv_path, self.data_loader.data_dir)
        else:
            self.validation_window.update_data_directory(self.data_loader.data_dir)
        self.validation_window.show()
        self.validation_window.raise_()  # Bring window to front

    def filter_files(self, search_text):
        """Filter files based on search text"""
        self.file_combo.clear()
        file_list = self.data_loader.get_file_list()
        
        if search_text:
            # Filter files that contain the search text
            filtered_files = [str(f) for f in file_list if str(f).find(search_text) >= 0]
        else:
            # If no search text, show all files
            filtered_files = [str(f) for f in file_list]
            
        self.file_combo.addItems(filtered_files)
        
        # Select first item if available
        if self.file_combo.count() > 0:
            self.load_file(self.file_combo.currentText())
    
    def save_filter_config(self):
        """Save current filter configuration"""
        try:
            # Ensure config directory exists
            os.makedirs(self.config_dir, exist_ok=True)
            
            config = {
                'low_cut': self.low_cut.value(),
                'high_cut': self.high_cut.value(),
                'filter_order': self.filter_order.value(),
                'last_saved': pd.Timestamp.now().isoformat()
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.statusBar().showMessage(f"Filter configuration saved to {self.config_file}", 3000)
            
        except PermissionError:
            self.statusBar().showMessage("Error: No tienes permisos para guardar la configuración", 5000)
        except Exception as e:
            self.statusBar().showMessage(f"Error saving configuration: {str(e)}", 5000)
    
    def load_filter_config(self):
        """Load saved filter configuration"""
        try:
            if not os.path.exists(self.config_file):
                self.statusBar().showMessage("No saved configuration found", 3000)
                return
                
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            # Update UI with loaded values
            self.low_cut.setValue(float(config['low_cut']))
            self.high_cut.setValue(float(config['high_cut']))
            self.filter_order.setValue(int(config['filter_order']))
            
            # Show last saved time if available
            if 'last_saved' in config:
                last_saved = pd.Timestamp(config['last_saved']).strftime('%Y-%m-%d %H:%M:%S')
                self.statusBar().showMessage(f"Loaded configuration from {last_saved}", 3000)
            else:
                self.statusBar().showMessage("Filter configuration loaded", 3000)
            
            # Apply filter if active
            if self.filter_active and self.current_data is not None:
                self.apply_filter()
            
        except json.JSONDecodeError:
            self.statusBar().showMessage("Error: El archivo de configuración está corrupto", 5000)
        except Exception as e:
            self.statusBar().showMessage(f"Error loading configuration: {str(e)}", 5000)
    
    def load_last_config(self):
        """Load last used configuration on startup"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                self.low_cut.setValue(float(config['low_cut']))
                self.high_cut.setValue(float(config['high_cut']))
                self.filter_order.setValue(int(config['filter_order']))
                
                # No mostrar mensaje al cargar la configuración inicial
                print("Loaded last used filter configuration")
        except Exception as e:
            print(f"Notice: Using default filter values - {str(e)}")
            # Use default values if loading fails
    
    def select_csv_file(self):
        """Open file selection dialog for CSV"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select P-wave Times CSV File",
            os.path.expanduser("~"),
            "CSV Files (*.csv)"
        )
        
        if file_path:
            success, message = self.data_loader.load_csv(file_path)
            if success:
                self.csv_path_label.setText(file_path)
                self.csv_path_label.setStyleSheet("color: green;")
                self.statusBar().showMessage("CSV file loaded successfully", 3000)
                
                # Reload current file if one is loaded
                if self.current_file_id is not None:
                    self.load_file(self.current_file_id)
            else:
                self.csv_path_label.setText("Error loading CSV")
                self.csv_path_label.setStyleSheet("color: red;")
                self.statusBar().showMessage(message, 5000)