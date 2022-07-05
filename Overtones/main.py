import os
import time
import numpy as np
from tqdm import tqdm # Progress bar
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.fft import irfft, rfft, rfftfreq

from utils import *


class FourierTransform:
    def __init__(self):
        pass

    def load_audio(self, filename, crop_start=0, crop_end=-1, chunk_sizes=0.02):
        self.filename = filename
        self.sample_rate, audio = wavfile.read(filename)
        if len(audio.shape) > 1:
            audio = audio[:,0] # Take only the first channel.
        audio = audio[int(crop_start * self.sample_rate):int(crop_end * self.sample_rate)]
        self.audio = audio/2**15 # normalize to: [-1,1)

        self.chunk_sizes = chunk_sizes if chunk_sizes != -1 else len(audio)/self.sample_rate

        self.frequencies = rfftfreq(int(self.chunk_sizes*self.sample_rate), 1/self.sample_rate)
        self.frequency_rounder = {f:round_frequency(f) for f in self.frequencies}
        # self.generate_envelope()
        self.split_audio(self.chunk_sizes)
    
    def generate_envelope(self, local_size=500):
        chunks = np.array_split(self.audio, int(len(self.audio)/local_size))
        self.audio = np.array([max(lst) for lst in chunks])
        return self.audio
    
    def split_audio(self, chunk_sizes=0.02):
        chunks = []
        for i in range(0, len(self.audio), int(chunk_sizes*self.sample_rate)):
            if i+int(chunk_sizes*self.sample_rate) > len(self.audio):
                break
            chunk = self.audio[i:i+int(chunk_sizes*self.sample_rate)]
            chunks.append(chunk)
        self.chunks = np.array(chunks)
        return self.chunks

    def get_rounded_frequencies(self):
        joined_freqs = []
        for i in range(len(self.frequencies)):
            rf = self.frequency_rounder[self.frequencies[i]]
            if len(joined_freqs) != 0 and joined_freqs[-1] == rf:
                continue
            joined_freqs.append(rf)
        return np.array(joined_freqs)
    

    def calculate_amplitudes(self):
        self.amplitudes = np.array([calc_rfft(chunk) for chunk in self.chunks])
        self.highest_amplitude = np.max(self.amplitudes)
        self.gradient = get_gradient(self.highest_amplitude)
        return self.amplitudes
    
    def apply_buckets(self, amplitudes, frequencies):
        new_amps = np.zeros_like(amplitudes)
        new_freqs = np.array([self.frequency_rounder[f] for f in frequencies])
        amp_index = 0
        for i, amp in enumerate(amplitudes):
            if i == 0:
                new_amps[0] = amp
            elif new_freqs[i-1] == new_freqs[i]:
                new_amps[amp_index] += amp
            else:
                amp_index = i
                new_amps[amp_index] = amp
        
        return new_amps, new_freqs
    

    def plot(self, amplitude_threshold=0.15, show_filtered=True):
        # print('Filtering...')
        filtered_amps_list = []
        filtered_freqs_list = []

        all_mults = []
        for i, amps in tqdm(enumerate(self.amplitudes), total=self.amplitudes.shape[0], disable=True):
            filtered_amps = amps[amps>amplitude_threshold*self.highest_amplitude]
            filtered_freqs = self.frequencies[amps>amplitude_threshold*self.highest_amplitude]
            # overtones = sorted(filtered_freqs, key=lambda x: filtered_amps[list(filtered_freqs).index(x)])

            mults = get_multiplies({f:filtered_amps[list(filtered_freqs).index(f)] for f in filtered_freqs})
            all_mults.append(mults)

            # new_overtones = []
            # for overtone in overtones:
            #     if any(abs(x - overtone) < 20 for x in new_overtones):
            #         continue
            #     new_overtones.append(overtone)
            # print(new_overtones)
            # print(len(new_overtones))


            # filtered_amps, filtered_freqs = self.apply_buckets(filtered_amps, filtered_freqs)
            # filtered_amps, filtered_freqs = filtered_amps[filtered_amps>0], filtered_freqs[filtered_amps>0]


            filtered_amps_list.append(filtered_amps)
            filtered_freqs_list.append(filtered_freqs)
        
        # print('Plotting...')

        # for i, (filtered_amps, filtered_freqs) in tqdm(enumerate(zip(filtered_amps_list, filtered_freqs_list)), total=len(filtered_amps_list)):
        #     color = self.gradient.to_rgba(filtered_amps)

        #     plt.scatter([i*self.chunk_sizes]*len(filtered_freqs), filtered_freqs, c=color, s=self.chunk_sizes*250, marker='s')
        
        for i, mult in enumerate(all_mults):
            mults = sorted(list(mult.keys()))
            amps = np.array([mult[m] for m in mults])
            color = self.gradient.to_rgba(amps)
            plt.scatter([i*self.chunk_sizes]*len(mults), mults, c=color, s=self.chunk_sizes*250, marker='s')

        ax = plt.gca()
        if show_filtered:
            ax.set_facecolor(self.gradient.to_rgba(0)) # paint background in 'zero color'.

        plt.colorbar(self.gradient, label='Amplitude')
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (S)")
        plt.title(self.filename)

        # plt.show()
        plt.savefig("results/" + self.filename.replace("/", ".").replace("\\", ".") + '.png')
        plt.clf()


def get_all_wavs(folder):
    wavs = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.wav'):
                name = os.path.join(root, file)
                if "guitar" in name or "keyboard" in name:
                    wavs.append(name)
    return wavs

if __name__ == "__main__":
    folder = 'audio/dataset'
    wavs = get_all_wavs(folder)

    for wav in tqdm(wavs):
        ft = FourierTransform()
        ft.load_audio(wav, crop_start=0, crop_end=-1, chunk_sizes=0.02)
        ft.calculate_amplitudes()
        ft.plot(0.1)