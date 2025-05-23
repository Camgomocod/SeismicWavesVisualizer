from obspy import read, UTCDateTime
import pandas as pd
import numpy as np
import os

class LabelValidator:
    def __init__(self, data, p_arrival_time, sampling_rate, start_time):
        self.data = data
        self.p_arrival_time = p_arrival_time
        self.sampling_rate = sampling_rate
        self.start_time = start_time
        self.duration = len(data) / sampling_rate
        self.end_time = start_time + self.duration
        
    def validate(self):
        """
        Validate P arrival time against signal duration
        Returns a dictionary with validation results
        """
        if self.p_arrival_time is None:
            return {
                'is_valid': False,
                'error': 'Missing P arrival time',
                'details': {
                    'signal_start': self.start_time.timestamp,
                    'signal_end': self.end_time.timestamp,
                    'duration': self.duration,
                    'p_arrival': None
                }
            }
        
        relative_p_time = self.p_arrival_time - self.start_time
        
        is_valid = 0 <= relative_p_time <= self.duration
        
        return {
            'is_valid': is_valid,
            'error': None if is_valid else 'P arrival time outside signal duration',
            'details': {
                'signal_start': self.start_time.timestamp,
                'signal_end': self.end_time.timestamp,
                'duration': self.duration,
                'p_arrival': self.p_arrival_time.timestamp,
                'relative_p_time': relative_p_time
            }
        }

class DataLoader:
    def __init__(self, csv_path=None, data_dir=None):
        """Initialize the data loader with optional CSV path and data directory"""
        self.data_dir = data_dir or os.getcwd()
        self.csv_path = csv_path
        self.df = None
        self.load_csv()
    
    def load_csv(self, csv_path=None):
        """Load CSV file with P-wave arrival times"""
        if csv_path:
            self.csv_path = csv_path
            
        if self.csv_path and os.path.exists(self.csv_path):
            try:
                self.df = pd.read_csv(self.csv_path)
                # Convert 'lec_p' column to numeric, setting invalid values to NaN
                self.df['lec_p'] = pd.to_numeric(self.df['lec_p'], errors='coerce')
                return True, "CSV loaded successfully"
            except Exception as e:
                self.df = pd.DataFrame(columns=['archivo', 'lec_p'])
                return False, f"Error reading CSV file: {str(e)}"
        else:
            self.df = pd.DataFrame(columns=['archivo', 'lec_p'])
            return False, "No CSV file specified or file not found"

    def set_data_directory(self, directory):
        """Set the directory where MSEED files are located"""
        self.data_dir = directory
    
    def get_mseed_path(self, file_id):
        """Get full path for a MSEED file"""
        matching_files = []
        
        # Search in directory and subdirectories
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if not file.endswith('.mseed'):
                    continue
                    
                # Extract base name without extension
                base_name = file.replace('.mseed', '')
                
                # Handle augmented files
                if '_aug' in base_name:
                    base_name = base_name.split('_aug')[0]
                
                # Try to match the ID (with or without leading zero)
                try:
                    current_id = int(base_name.lstrip('0'))
                    if current_id == int(file_id):
                        matching_files.append(os.path.join(root, file))
                except ValueError:
                    continue
        
        # If we found matches, return the first one
        # (You might want to implement a different selection strategy)
        if matching_files:
            return matching_files[0]
            
        # If no matches found, try the old way for backward compatibility
        direct_path = os.path.join(self.data_dir, f"{file_id}.mseed")
        if os.path.exists(direct_path):
            return direct_path
            
        zero_path = os.path.join(self.data_dir, f"0{file_id}.mseed")
        return zero_path  # Return this path even if it doesn't exist (backward compatibility)
    
    def load_mseed(self, file_name):
        """
        Load MSEED file and return the trace data and metadata
        """
        try:
            # Read the MSEED file
            st = read(file_name)
            if len(st) == 0:
                raise ValueError("No traces found in MSEED file")
            
            tr = st[0]
            
            # Get signal data and normalize it safely
            data = tr.data
            if len(data) == 0:
                raise ValueError("Empty trace data")
                
            # Safe normalization with handling for zero standard deviation
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val == 0:
                data_normalized = data - mean_val  # Just center if no variation
            else:
                data_normalized = (data - mean_val) / std_val
            
            # Setup time information
            start_time = tr.stats.starttime
            sampling_rate = tr.stats.sampling_rate
            if sampling_rate <= 0:
                raise ValueError(f"Invalid sampling rate: {sampling_rate}")
                
            npts = tr.stats.npts
            times = np.arange(0, npts) / sampling_rate
            
            return {
                'data': data,
                'data_normalized': data_normalized,
                'times': times,
                'sampling_rate': sampling_rate,
                'start_time': start_time,
                'stats': tr.stats
            }
            
        except Exception as e:
            error_msg = f"Error loading MSEED file {file_name}: {str(e)}"
            print(error_msg)
            # Re-raise with more context
            raise IOError(error_msg) from e
    
    def get_p_arrival(self, file_id):
        """
        Get P-wave arrival time for a given file ID
        Returns None if no valid P arrival time exists
        """
        # Convert file_id to integer to handle both string and int inputs
        try:
            file_id = int(str(file_id).lstrip('0'))  # Remove leading zeros if any
        except ValueError:
            return None
            
        if file_id not in self.df['archivo'].values:
            return None
            
        p_time = self.df.loc[self.df['archivo'] == file_id, 'lec_p'].values
        if len(p_time) > 0 and not pd.isna(p_time[0]):
            return UTCDateTime(float(p_time[0]))
        return None
    
    def validate_labels(self, file_id, signal_data):
        """
        Validate labels for a given file
        Handles cases with missing P arrival times
        """
        # Calculate signal duration
        duration = len(signal_data['data']) / signal_data['sampling_rate']
        
        validation_result = {
            'is_valid': True,
            'error': None,
            'details': {
                'file_id': file_id,
                'duration': duration,
                'p_arrival': None,
                'relative_p_time': None,
                'has_p_arrival': False
            }
        }
        
        # Get P arrival time
        p_arrival = self.get_p_arrival(file_id)
        
        if p_arrival is None:
            validation_result.update({
                'is_valid': True,  # Consider it valid but mark as no P arrival
                'error': 'No P arrival time available',
                'details': {
                    'file_id': file_id,
                    'duration': duration,
                    'p_arrival': None,
                    'relative_p_time': None,
                    'has_p_arrival': False
                }
            })
            return validation_result
        
        # If we have a P arrival time, validate it
        validation_result['details']['has_p_arrival'] = True
        relative_p_time = p_arrival - signal_data['start_time']
        validation_result['details'].update({
            'p_arrival': p_arrival.timestamp,
            'relative_p_time': relative_p_time
        })
        
        # Check if P arrival is within signal duration
        if relative_p_time < 0 or relative_p_time > duration:
            validation_result.update({
                'is_valid': False,
                'error': f'P arrival time ({relative_p_time:.2f}s) outside signal duration (0 to {duration:.2f}s)'
            })
        
        return validation_result
    
    def get_file_list(self):
        """
        Return list of available file IDs
        """
        mseed_files = []
        # Scan directory and subdirectories for MSEED files
        try:
            for root, dirs, files in os.walk(self.data_dir):
                for file in files:
                    if file.endswith('.mseed'):
                        try:
                            # Extract base filename without extension
                            base_name = file.replace('.mseed', '')
                            
                            # Handle augmented files (with _aug suffix)
                            if '_aug' in base_name:
                                # Extract the ID part before _aug
                                base_name = base_name.split('_aug')[0]
                            
                            # Remove leading zeros and convert to integer
                            file_id = int(base_name.lstrip('0'))
                            if file_id not in mseed_files:
                                mseed_files.append(file_id)
                        except ValueError:
                            print(f"Warning: Could not parse file ID from {file}")
                            continue
        except Exception as e:
            print(f"Error scanning directory: {e}")
        
        return sorted(mseed_files)