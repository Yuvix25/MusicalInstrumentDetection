import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.fft import irfft, rfft, rfftfreq
from scipy.io import wavfile
import scipy.signal as sps


def downsample(audio, rate, new_rate):
    number_of_samples = round(len(audio) * float(new_rate) / rate)
    return new_rate, sps.resample(audio, number_of_samples)



def filename_to_instrument(filename):
    return '.'.join(filename.replace('\\', '/').split('/')[-1].split('_')[:2])

def load_wav(path, start=0, end=-1):
    rate, audio = wavfile.read(path)
    rate, audio = downsample(audio, rate, 16000)
    if len(audio.shape) > 1:
        audio = audio[:,0] # Take only the first channel.
    audio = audio[int(start * rate):int(end * rate)]
    audio = audio/2**15 # normalize to: [-1,1)
    return rate, audio

def preprocess(sample_rate, wav_data, window_size=0.02, amplitude_thresh=0.15):
    """
        - sample_rate: sample rate of the wav file
        - wav_data: numpy array of audio data
        - window_size: size of window in seconds
        - amplitude_thresh: threshold for amplitude
    """
    windows = np.array_split(wav_data, len(wav_data) // int(sample_rate * window_size))
    fft = np.array(list(map(calc_rfft, windows)))
    fft = filtered_fft(sample_rate, fft, amplitude_thresh) # [{freq:amp}]
    return fft

def filtered_fft(sr, window_amplitudes, threshold=0.15):
    thresh = np.max(window_amplitudes) * threshold
    return [{freq:(amp if amp > thresh else 0) for freq, amp in zip(rfftfreq(len(amps), 1/sr), amps)} for amps in window_amplitudes]




BASE_NOTE_FREQ = 110
BUCKET_COUNT = 12*8 # 12 semitones in an octave (semitone = 1/2 tone), and 1/16 of a tone

def round_frequency(frequency):
    if frequency == 0:
        return 0

    if frequency < BASE_NOTE_FREQ:
        rnd = lambda x: int(abs(x)+1) * x/abs(x)
    else:
        rnd = lambda x: int(x)

    rounded_with_base = BASE_NOTE_FREQ * 2**rnd(math.log2(frequency / BASE_NOTE_FREQ))
    bucket_index = np.floor(math.log(frequency / rounded_with_base, 2**(1/BUCKET_COUNT)))

    return rounded_with_base * (2**(1/BUCKET_COUNT))**bucket_index



def calc_rfft(audio):
    fft = rfft(audio)
    return np.abs(fft)



def get_gradient(max_num, name='viridis'):
    cmap = plt.get_cmap(name)
    c_norm = colors.Normalize(vmin=0, vmax=max_num)
    return cmx.ScalarMappable(norm=c_norm, cmap=cmap)



def get_multiplies(frequencies, freq_thresh=15, mult_thersh=0.25):
    if (len(frequencies) == 0):
        return {}
    
    groups = dict()
    for freq in frequencies:
        for (freq_avg, amp_avg) in groups:
            if abs(freq - freq_avg) < freq_thresh:
                new_avg = (sum(groups[freq_avg]) + freq) / (len(groups[freq_avg]) + 1)
                new_amp_avg = (sum(map(lambda x: frequencies[x], groups[freq_avg])) + frequencies[freq]) / (len(groups[freq_avg]) + 1)
                groups[new_avg, new_amp_avg] = groups[freq_avg, amp_avg] + [freq]
                if freq_avg != new_avg: del groups[freq_avg, amp_avg]
                break
        else:
            groups[freq, frequencies[freq]] = [freq]

    # filter 0
    groups = {k:v for k,v in groups.items() if k[0] != 0}
    if len(groups) == 0:
        return {}

    # min_freq = min(map(lambda x: x[0], groups.keys()))
    # freq with max amp
    base_freq = max(groups.keys(), key=lambda x: x[1])[0]
    mults = {}
    for (freq_avg, amp_avg) in groups:
        mults[freq_avg / base_freq] = amp_avg

    new_mults = {}
    for mult in mults:
        if abs(mult-round(mult)) < mult_thersh:
            if round(mult) not in new_mults:
                new_mults[round(mult)] = mults[mult]
    
    return new_mults