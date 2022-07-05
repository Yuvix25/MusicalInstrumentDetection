import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.fft import irfft, rfft, rfftfreq



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