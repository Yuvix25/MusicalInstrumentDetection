from threading import local
import numpy as np
from tqdm import tqdm # Progress bar
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from utils import *

class ADSRDetector:
    def __init__(self) -> None:
        pass

    def load_audio(self, filename, crop_start=0, crop_end=-1):
        self.sample_rate, audio = wavfile.read(filename)
        if len(audio.shape) > 1:
            audio = audio[:,0] # Take only the first channel.
        audio = audio[int(crop_start * self.sample_rate):-1 if crop_end == -1 else int(crop_end * self.sample_rate)]
        self.audio = audio/2**15 # normalize to: [-1,1)
        self.generate_envelope()
        self.smoothen_envelope()


    def generate_envelope(self, local_size=500):
        chunks = np.array_split(self.audio, int(len(self.audio)/local_size))
        self.envelope = np.array([max(lst) for lst in chunks])
        return self.envelope

    def smoothen_envelope(self):
        func = function_from_series(self.envelope)
        self.poly = taylor_approx(func, len(self.envelope))
    
    def get_amps(self):
        chunk_sizes = 0.2
        chunks = np.array_split(self.audio, int(len(self.audio)/(self.sample_rate*chunk_sizes)))
        fft = np.array([abs(np.fft.rfft(chunk)) for chunk in chunks])
        ret = {(int(self.sample_rate*chunk_sizes*i), int(self.sample_rate*chunk_sizes*(i+1))): max(fft[i]) for i in range(len(fft))}
        ret[-self.sample_rate*chunk_sizes, 0] = 0
        return ret

    def get_adsr(self, slope_threshold=0.005, release_low_threshold=0.05, release_stab_threshold=30, ultra_low_release_threshold=5):
        amps = self.get_amps()
        adsr = []
        lst = sorted(list(amps.keys()), key=lambda x: x[0])
        prev = amps[lst[0]]

        started_attack = False
        possibly_release = -1
        prev_slope_change_amp = None
        prev_slope = None
        for i in range(1, len(lst)):
            amp = amps[lst[i]]
            slope = (amp - prev)/(lst[i][0] - lst[i-1][0])

            if len(adsr) >= 1:
                low = release_low_threshold * amps[lst[adsr[0]]]
            # attack:
            if len(adsr) == 0:
                if slope > slope_threshold:
                    started_attack = True
                
                if slope < slope_threshold and started_attack:
                    adsr.append(i-1) # last index of the attack "high point"
            
            # decay:
            elif len(adsr) == 1:
                if slope > -slope_threshold:
                    adsr.append(i-1) # last index of the decay "low after high point"

            elif len(adsr) == 2:
                # undo decay:
                if slope > slope_threshold and (amp - amps[lst[adsr[0]]]) / (lst[i][0] - lst[adsr[0]][0]) > -slope_threshold:
                    adsr[1] = adsr[0]
                # sustain
                elif slope < -slope_threshold and possibly_release == -1:
                    possibly_release = i - 1
                    # adsr.append(i-1) # last index of the sustain
            
            # release
            # elif len(adsr) == 3:
                if amp < low and possibly_release != -1:
                    adsr.append(possibly_release)
                elif slope > -slope_threshold/2 and amp >= low:
                    possibly_release = -1
            
            if len(adsr) >= 2:
                if amp < low and prev_slope != None and prev_slope * slope <= 0:
                    if prev_slope_change_amp != None and abs(prev_slope_change_amp - amp) <= release_stab_threshold:
                        if len(adsr) == 2:
                            adsr.append(adsr[1])
                        adsr.append(i)
                        break
                        
                    prev_slope_change_amp = amp
                elif amp < ultra_low_release_threshold:
                    if len(adsr) == 2:
                        adsr.append(adsr[1])
                    adsr.append(i)
                    break


            prev = amp
            prev_slope = slope
        
        adsr = [lst[i][0] for i in adsr]
        print(adsr)
        return adsr


    def plot(self):
        # plt.plot(self.smoothened_envelope)
        print(self.poly.coefficients)
        plt.plot(self.envelope)

        amps = self.get_amps()
        xs = []
        ys = []
        for (s, e) in sorted(list(amps.keys()), key=lambda x: x[0]):
            xs.append(s)
            ys.append(amps[s, e])
        plt.plot(xs, ys, marker='o')

        for x in self.get_adsr():
            plt.axvline(x)
        # plt.plot(self.audio*600)

        # plt.plot([self.poly(x) for x in range(len(self.envelope))])
        plt.plot()
        # plt.show()
        plt.show()
    

if __name__ == "__main__":
    adsr = ADSRDetector()
    adsr.load_audio("audio/piano/notes.wav")
    # adsr.load_audio("audio/violin/example.wav")
    # adsr.load_audio("audio/guitar/G.wav")
    adsr.plot()

