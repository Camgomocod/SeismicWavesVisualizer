from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                          QPushButton, QTableWidget, QTableWidgetItem, QLabel,
                          QHeaderView, QProgressBar, QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from src.data.data_loader import DataLoader
import os
import pandas as pd

class ValidationWindow(QMainWindow):
    def __init__(self, csv_path, data_dir=None):
        super().__init__()
        self.setWindowTitle("Label Validation Analysis")
        self.resize(800, 600)
        
        # Initialize data loader
        self.data_loader = DataLoader(csv_path, data_dir)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add data directory controls
        data_controls = QHBoxLayout()
        self.data_dir_label = QLabel("Data Directory:")
        self.data_dir_path = QLabel(self.data_loader.data_dir)
        self.select_dir_button = QPushButton("Select Directory")
        self.select_dir_button.clicked.connect(self.select_data_directory)
        
        data_controls.addWidget(self.data_dir_label)
        data_controls.addWidget(self.data_dir_path)
        data_controls.addWidget(self.select_dir_button)
        data_controls.addStretch()
        
        layout.addLayout(data_controls)
        
        # Add header with controls
        header_layout = QHBoxLayout()
        self.status_label = QLabel("Click 'Analyze Files' to start validation")
        header_layout.addWidget(self.status_label)
        
        # Add analyze button
        self.analyze_button = QPushButton("Analyze Files")
        self.analyze_button.clicked.connect(self.start_analysis)
        header_layout.addWidget(self.analyze_button)
        
        # Add export button
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        header_layout.addWidget(self.export_button)
        
        layout.addLayout(header_layout)
        
        # Add progress bar (hidden initially)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        # Create table for results
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "File ID", 
            "Status",
            "Signal Duration (s)",
            "P Arrival Time (s)",
            "Details"
        ])
        
        # Configure table
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        
        layout.addWidget(self.table)
        
        # Add summary label
        self.summary_label = QLabel()
        layout.addWidget(self.summary_label)
        
        # Setup timer for progress updates
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.current_progress = 0
        self.analyzed_files = 0
        self.total_files = 0
        
        # Store validation results
        self.validation_results = []
    
    def update_data_directory(self, directory):
        """Update the data directory"""
        self.data_loader.set_data_directory(directory)
        self.data_dir_path.setText(directory)
    
    def select_data_directory(self):
        """Open directory selection dialog"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Data Directory",
            self.data_loader.data_dir,
            QFileDialog.ShowDirsOnly
        )
        
        if directory:
            self.update_data_directory(directory)
    
    def start_analysis(self):
        """Start the analysis process"""
        self.status_label.setText("Analyzing files...")
        self.analyze_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.table.setRowCount(0)
        self.summary_label.clear()
        self.validation_results = []
        
        # Show and reset progress bar
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.current_progress = 0
        self.analyzed_files = 0
        
        # Get file list and start analysis
        self.file_list = self.data_loader.get_file_list()
        self.total_files = len(self.file_list)
        
        # Start progress timer
        self.progress_timer.start(50)  # Update every 50ms
        
        # Start actual analysis
        QTimer.singleShot(100, self.analyze_next_file)
    
    def analyze_next_file(self):
        """Analyze the next file in the list"""
        if self.analyzed_files >= len(self.file_list):
            self.finish_analysis()
            return
        
        file_id = self.file_list[self.analyzed_files]
        try:
            file_path = self.data_loader.get_mseed_path(file_id)
            if os.path.exists(file_path):
                # Load and validate file
                signal_data = self.data_loader.load_mseed(file_path)
                validation_result = self.data_loader.validate_labels(file_id, signal_data)
                self.validation_results.append(validation_result)
                
                if not validation_result['is_valid']:
                    details = validation_result['details']
                    
                    # Add row to table
                    row = self.table.rowCount()
                    self.table.insertRow(row)
                    
                    # File ID
                    self.table.setItem(row, 0, QTableWidgetItem(str(file_id)))
                    
                    # Status
                    status_item = QTableWidgetItem("Invalid")
                    status_item.setForeground(QColor('red'))
                    self.table.setItem(row, 1, status_item)
                    
                    # Signal Duration
                    duration_item = QTableWidgetItem(f"{details['duration']:.2f}")
                    self.table.setItem(row, 2, duration_item)
                    
                    # P Arrival Time
                    if details['p_arrival'] is None:
                        p_time_item = QTableWidgetItem("Missing")
                    else:
                        relative_time = details['relative_p_time']
                        p_time_item = QTableWidgetItem(f"{relative_time:.2f}")
                    self.table.setItem(row, 3, p_time_item)
                    
                    # Error Details
                    self.table.setItem(row, 4, QTableWidgetItem(validation_result['error']))
        
        except Exception as e:
            print(f"Error analyzing file {file_id}: {e}")
        
        self.analyzed_files += 1
        
        # Schedule next file analysis
        QTimer.singleShot(1, self.analyze_next_file)
    
    def update_progress(self):
        """Update the progress bar animation"""
        if self.total_files > 0:
            progress = (self.analyzed_files * 100) // self.total_files
            self.progress_bar.setValue(progress)
    
    def finish_analysis(self):
        """Finish the analysis process"""
        self.progress_timer.stop()
        self.progress_bar.hide()
        self.analyze_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.status_label.setText("Analysis complete")
        
        # Update summary
        invalid_count = self.table.rowCount()
        self.summary_label.setText(
            f"Found {invalid_count} files with invalid labels out of {self.total_files} total files "
            f"({invalid_count/self.total_files*100:.1f}%)"
        )
    
    def export_results(self):
        """Export validation results to CSV"""
        if not self.validation_results:
            return
            
        # Ask for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Validation Results",
            os.path.join(os.path.expanduser("~"), "validation_results.csv"),
            "CSV Files (*.csv)"
        )
        
        if not file_path:
            return
            
        try:
            # Create DataFrame from results
            data = []
            for result in self.validation_results:
                details = result['details']
                data.append({
                    'File ID': details['file_id'],
                    'Is Valid': result['is_valid'],
                    'Signal Duration (s)': details['duration'],
                    'P Arrival Time (s)': details['relative_p_time'] if details['p_arrival'] else None,
                    'Error': result['error'] if not result['is_valid'] else None
                })
            
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            
            self.status_label.setText(f"Results exported to {file_path}")
        
        except Exception as e:
            self.status_label.setText(f"Error exporting results: {str(e)}")