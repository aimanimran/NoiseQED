import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import speech_recognition as sr
from scipy.signal import wiener
import pywt

def speech_intelligibility_score(audio_path):

    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            
        # Transcribe
        try:
            transcript = recognizer.recognize_google(audio)
            return len(transcript)
        except sr.UnknownValueError:
            return 0
        except sr.RequestError:
            return -1
    except Exception as e:
        print(f"Error processing audio: {e}")
        return -1

def noise_analysis(input_path):
    # Load the audio file
    y, sr = librosa.load(input_path, sr=None)
    
    def noise_reduction_techniques(y):
        techniques = {
            'Original': original,
            'Spectral Subtraction': spectral_subtraction,
            'Wiener Filter': wiener_filtering,
            'Soft Thresholding': soft_thresholding,
            'Wavelet Denoising': wavelet_denoising,
            'Rolling Window Noise Reduction': rolling_window_noise_reduction
        }
        return {name: technique(y, sr) for name, technique in techniques.items()}
    
    def original(y, sr):
        return y
    
    def spectral_subtraction(y, sr):
        D = librosa.stft(y)
        magnitude = np.abs(D)
        noise_floor = np.percentile(magnitude, 10, axis=1)
        reduced_magnitude = np.maximum(magnitude - noise_floor[:, np.newaxis], 0)
        return librosa.istft(reduced_magnitude * np.exp(1j * np.angle(D)))
    
    def wiener_filtering(y, sr):
        return wiener(y, 5)
    
    def soft_thresholding(y, sr):
        D = librosa.stft(y)
        magnitude = np.abs(D)
        threshold = np.median(magnitude) * 1.5
        reduced_magnitude = np.where(magnitude > threshold, magnitude, 0)
        return librosa.istft(reduced_magnitude * np.exp(1j * np.angle(D)))
    
    def wavelet_denoising(y, sr):
        coeffs = pywt.wavedec(y, 'db4', level=5)
        
        threshold = np.sqrt(2 * np.log(len(y)))
        new_coeffs = []
        for i, coeff in enumerate(coeffs):
            if i == 0:
                new_coeffs.append(coeff)
            else:
                new_coeffs.append(pywt.threshold(coeff, threshold, mode='soft'))
        
        return pywt.waverec(new_coeffs, 'db4')
    
    def rolling_window_noise_reduction(y, sr, window_size=0.025):
        window_length = int(window_size * sr)
        cleaned_signal = np.zeros_like(y)
        
        for i in range(0, len(y), window_length):
            window = y[i:i+window_length]
            if len(window) < window_length:
                break
            
            noise_threshold = np.median(np.abs(window)) * 1.5
            
            cleaned_window = np.where(np.abs(window) > noise_threshold, window, 0)
            cleaned_signal[i:i+window_length] = cleaned_window
        
        return cleaned_signal
    
    # Apply noise reduction techniques
    reduced_audios = noise_reduction_techniques(y)
    
    # Evaluate scores
    technique_scores = {}
    technique_transcripts = {} #AA
    for name, processed_y in reduced_audios.items():

        # Temporarily save processed audio for evaluation
        temp_path = f'{name}_denoised.wav'
        sf.write(temp_path, processed_y, sr)
        
        score = speech_intelligibility_score(temp_path)
        technique_scores[name] = score
    
    best_technique = max(technique_scores, key=technique_scores.get)
    best_audio = reduced_audios[best_technique]
    
    sf.write('best_denoised.wav', best_audio, sr)
    
    # Visualization (optional)
    plt.figure(figsize=(15, 10))
    
    plot_positions = [2, 3, 4, 5, 6]
    for pos, (name, processed_y) in zip(plot_positions, list(reduced_audios.items())):
        plt.subplot(2, 3, pos)
        D_processed = librosa.stft(processed_y)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_processed), ref=np.max), 
                                sr=sr, y_axis='hz', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'{name} (Score: {technique_scores[name]})')

    plt.tight_layout()
    plt.savefig('noise_reduction_comparison.png')
    plt.close()
    
    print("Technique Scores:", technique_scores)
    print(f"Best Technique: {best_technique}")
    
    return best_audio, reduced_audios

# Example usage
input_audio = sample_noisy.wav
best_audio, all_processed_audios = noise_analysis(input_audio)
