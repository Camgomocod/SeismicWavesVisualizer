import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging except errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only

import tensorflow as tf
import numpy as np
import pywt
from scipy import signal as sig

# Configure TensorFlow for CPU
tf.config.set_visible_devices([], 'GPU')

class PWavePredictor:
    def __init__(self, model_path=None):
        """Initialize the P-wave arrival time predictor"""
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                    'models', 'augmeted_model.keras')
        self.model = tf.keras.models.load_model(model_path)
    
    def pad_or_trim(self, signal, target_length=8000):
        """Pad or trim signal to target length while preserving signal characteristics"""
        signal = np.array(signal, dtype=np.float32)
        
        if len(signal) > target_length:
            # Find zero crossings to avoid cutting in middle of wave
            zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
            if len(zero_crossings) > 0:
                # Find closest zero crossing to target length
                cut_point = zero_crossings[np.abs(zero_crossings - target_length).argmin()]
                cut_point = min(cut_point, target_length)
                return signal[:cut_point]
            return signal[:target_length]
        elif len(signal) < target_length:
            # Calculate padding
            pad_width = target_length - len(signal)
            # Use edge padding to avoid discontinuities
            return np.pad(signal, (0, pad_width), mode='edge')
        return signal
    
    def preprocess_signal(self, signal_data):
        """Preprocess the signal maintaining its characteristics"""
        # Convert to numpy array if needed
        signal = np.array(signal_data, dtype=np.float32)
        
        # Remove DC offset
        signal = signal - np.mean(signal)
        
        # Normalize amplitude while preserving relative amplitudes
        max_amp = np.max(np.abs(signal))
        if (max_amp > 0):
            signal = signal / max_amp
        
        # Apply bandpass filter to remove noise while preserving signal shape
        nyquist = 50  # Assuming 100Hz sampling rate
        low = 1.0 / nyquist
        high = 20.0 / nyquist
        b, a = sig.butter(4, [low, high], btype='band')
        signal = sig.filtfilt(b, a, signal)
        
        # Pad or trim to target length
        signal = self.pad_or_trim(signal)
        
        return signal.reshape(-1, 1)
    
    def compute_wavelet_features(self, signal, wavelet='db4', level=5):
        """Compute wavelet features matching the training process"""
        # Ensure signal is preprocessed
        signal = signal.flatten()
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
        
        # Extract features for each level
        features = []
        for i, coeff in enumerate(coeffs):
            # Statistical features
            features.extend([
                np.mean(coeff),
                np.std(coeff),
                np.max(coeff),
                np.min(coeff),
                np.median(coeff),
                np.sum(np.abs(coeff)),  # L1 norm
                np.sqrt(np.sum(coeff**2)),  # L2 norm
                np.mean(np.abs(coeff)),  # Mean absolute value
                np.var(coeff),  # Variance
                np.percentile(coeff, 75)  # 75th percentile
            ])
            
            if i < len(coeffs) - 1:
                # Frequency domain features for detail coefficients
                freq_features = np.abs(np.fft.fft(coeff))[:len(coeff)//2]
                features.extend([
                    np.mean(freq_features),
                    np.max(freq_features)
                ])
        
        # Ensure we have exactly 60 features
        features = np.array(features, dtype=np.float32)
        if len(features) < 60:
            features = np.pad(features, (0, 60 - len(features)), mode='constant')
        else:
            features = features[:60]
        
        return features.reshape(1, -1)
    
    def predict(self, signal_data, sampling_rate):
        """
        Predict P-wave arrival time for a signal
        
        Args:
            signal_data: Raw signal data
            sampling_rate: Sampling rate in Hz
            
        Returns:
            predicted_time: Predicted P arrival time in seconds
        """
        try:
            # Preprocess signal preserving waveform characteristics
            processed_signal = self.preprocess_signal(signal_data)
            
            # Extract wavelet features
            wavelet_features = self.compute_wavelet_features(processed_signal)
            
            # Verify shapes
            assert processed_signal.shape[0] <= 8000, f"Signal too long: {processed_signal.shape}"
            assert wavelet_features.shape == (1, 60), f"Feature shape mismatch: {wavelet_features.shape}"
            
            # Make prediction
            prediction = self.model.predict([
                processed_signal.reshape(1, -1, 1),
                wavelet_features
            ], verbose=0)
            
            return float(prediction[0][0])
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise