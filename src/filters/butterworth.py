import numpy as np
from scipy import signal

def validate_filter_params(fs, lowcut, highcut, order):
    """Validate filter parameters and return normalized frequencies"""
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}")
    
    if order < 1:
        raise ValueError(f"Filter order must be at least 1, got {order}")
    
    # Ensure frequencies are positive
    if lowcut <= 0 or highcut <= 0:
        raise ValueError(f"Cutoff frequencies must be positive, got lowcut={lowcut}, highcut={highcut}")
    
    # Check Nyquist criterion
    nyq = 0.5 * fs
    if lowcut >= nyq or highcut >= nyq:
        raise ValueError(f"Cutoff frequencies must be below Nyquist frequency ({nyq}Hz)")
    
    # Normalize frequencies to Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    
    # Ensure low < high and within valid range
    low = np.clip(low, 0.001, 0.99)
    high = np.clip(high, 0.001, 0.99)
    
    if low >= high:
        raise ValueError(f"Low cutoff must be less than high cutoff, got {lowcut}Hz and {highcut}Hz")
    
    return low, high

def bandpass_filter(signal_data, fs, lowcut=1.0, highcut=20.0, order=4):
    """
    Apply a bandpass Butterworth filter to the signal.
    
    Args:
        signal_data: Input signal (numpy array)
        fs: Sampling frequency in Hz
        lowcut: Lower cutoff frequency in Hz
        highcut: Higher cutoff frequency in Hz
        order: Filter order
        
    Returns:
        Filtered signal with same scale as input
        
    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If filtering fails
    """
    try:
        # Input validation
        if not isinstance(signal_data, np.ndarray):
            signal_data = np.array(signal_data)
        
        if signal_data.size == 0:
            raise ValueError("Empty input signal")
        
        # Validate and normalize frequencies
        low, high = validate_filter_params(fs, lowcut, highcut, order)
        
        # Create filter
        try:
            b, a = signal.butter(order, [low, high], btype='band')
        except Exception as e:
            raise RuntimeError(f"Failed to create Butterworth filter: {str(e)}")
        
        # Apply filter
        try:
            filtered = signal.filtfilt(b, a, signal_data)
        except Exception as e:
            raise RuntimeError(f"Failed to apply filter: {str(e)}")
        
        # Scale filtered signal to match input amplitude
        input_std = np.std(signal_data)
        if input_std > 0:  # Only scale if input has variation
            filtered_std = np.std(filtered)
            if filtered_std > 0:  # Avoid division by zero
                scale_factor = input_std / filtered_std
                filtered = filtered * scale_factor
        
        return filtered
        
    except Exception as e:
        # Add context to any unhandled exceptions
        raise RuntimeError(f"Error in bandpass filter: {str(e)}") from e