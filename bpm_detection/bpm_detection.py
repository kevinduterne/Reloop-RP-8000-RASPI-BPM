# Copyright 2012 Free Software Foundation, Inc.
#
# This file is part of The BPM Detector Python
#
# The BPM Detector Python is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# The BPM Detector Python is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with The BPM Detector Python; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.

import argparse
import array
import math
import wave

import matplotlib.pyplot as plt
import numpy
import pywt
from scipy import signal


from ctypes import *
from contextlib import contextmanager
import pyaudio
import re

from rp8000 import midi
import rtmidi
import time

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)


def read_audio(window):
    form_1 = pyaudio.paInt32
    chans = 1 # 1 channel
    samp_rate = 44100 # 44.1kHz sampling rate
    chunk = 4096 # 2^12 samples for buffer
    record_secs = window # seconds to record
    dev_index = 1 # device index found by p.get_device_info_by_index(ii)
    wav_output_filename = filename # name of .wav file
    
    with noalsaerr():
        audio = pyaudio.PyAudio() # create pyaudio instantiation
    
    # create pyaudio stream
    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                    input_device_index = dev_index,input = True, \
                    frames_per_buffer=chunk)
    print("recording")
    frames = []
    
    # loop through stream and append audio chunks to frame array
    for ii in range(0,int((samp_rate/chunk)*record_secs)):
        data = stream.read(chunk)
        frames.append(data)
    
    print("finished recording")
    
    # stop the stream, close it, and terminate the pyaudio instantiation
    stream.stop_stream()
    stream.close()
    audio.terminate()    

    # typ = choose_type( wf.getsampwidth() ) # TODO: implement choose_type
    nsamps = len(frames)
    assert nsamps > 0

    fs = samp_rate
    assert fs > 0

    # Read entire file and make into an array
    samps = frames 

    try:
        assert nsamps == len(samps)
    except AssertionError:
        print(nsamps, "not equal to", len(samps))

    return samps, fs


def record_wav(filename,window):
    form_1 = pyaudio.paInt32
    chans = 1 # 1 channel
    samp_rate = 44100 # 44.1kHz sampling rate
    chunk = 4096 # 2^12 samples for buffer
    record_secs = window # seconds to record
    dev_index = 1 # device index found by p.get_device_info_by_index(ii)
    wav_output_filename = filename # name of .wav file
    
    with noalsaerr():
        audio = pyaudio.PyAudio() # create pyaudio instantiation
    
    # create pyaudio stream
    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                    input_device_index = dev_index,input = True, \
                    frames_per_buffer=chunk)
    print("recording")
    frames = []
    
    # loop through stream and append audio chunks to frame array
    for ii in range(0,int((samp_rate/chunk)*record_secs)):
        data = stream.read(chunk)
        frames.append(data)
    
    print("finished recording")
    
    # stop the stream, close it, and terminate the pyaudio instantiation
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # save the audio frames as .wav file
    wavefile = wave.open(wav_output_filename,'wb')
    wavefile.setnchannels(chans)
    wavefile.setsampwidth(audio.get_sample_size(form_1))
    wavefile.setframerate(samp_rate)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()



def read_wav(filename):
    # open file, get metadata for audio
    try:
        wf = wave.open(filename, "rb")
    except IOError as e:
        print(e)
        return

    # typ = choose_type( wf.getsampwidth() ) # TODO: implement choose_type
    nsamps = wf.getnframes()
    assert nsamps > 0

    fs = wf.getframerate()
    assert fs > 0

    # Read entire file and make into an array
    samps = list(array.array("i", wf.readframes(nsamps)))

    try:
        assert nsamps == len(samps)
    except AssertionError:
        print(nsamps, "not equal to", len(samps))

    return samps, fs


# print an error when no data can be found
def no_audio_data():
    print("No audio data for sample, skipping...")
    return None, None


# simple peak detection
def peak_detect(data):
    max_val = numpy.amax(abs(data))
    peak_ndx = numpy.where(data == max_val)
    if len(peak_ndx[0]) == 0:  # if nothing found then the max must be negative
        peak_ndx = numpy.where(data == -max_val)
    return peak_ndx


def bpm_detector(data, fs):
    cA = []
    cD = []
    correl = []
    cD_sum = []
    levels = 4
    max_decimation = 2 ** (levels - 1)
    min_ndx = math.floor(60.0 / 220 * (fs / max_decimation))
    max_ndx = math.floor(60.0 / 40 * (fs / max_decimation))

    for loop in range(0, levels):
        cD = []
        # 1) DWT
        if loop == 0:
            [cA, cD] = pywt.dwt(data, "db4")
            cD_minlen = len(cD) / max_decimation + 1
            cD_sum = numpy.zeros(math.floor(cD_minlen))
        else:
            [cA, cD] = pywt.dwt(cA, "db4")

        # 2) Filter
        cD = signal.lfilter([0.01], [1 - 0.99], cD)

        # 4) Subtract out the mean.

        # 5) Decimate for reconstruction later.
        cD = abs(cD[:: (2 ** (levels - loop - 1))])
        cD = cD - numpy.mean(cD)

        # 6) Recombine the signal before ACF
        #    Essentially, each level the detail coefs (i.e. the HPF values) are concatenated to the beginning of the array
        cD_sum = cD[0 : math.floor(cD_minlen)] + cD_sum

    if [b for b in cA if b != 0.0] == []:
        return no_audio_data()

    # Adding in the approximate data as well...
    cA = signal.lfilter([0.01], [1 - 0.99], cA)
    cA = abs(cA)
    cA = cA - numpy.mean(cA)
    cD_sum = cA[0 : math.floor(cD_minlen)] + cD_sum

    # ACF
    correl = numpy.correlate(cD_sum, cD_sum, "full")

    midpoint = math.floor(len(correl) / 2)
    correl_midpoint_tmp = correl[midpoint:]
    peak_ndx = peak_detect(correl_midpoint_tmp[min_ndx:max_ndx])
    if len(peak_ndx) > 1:
        return no_audio_data()

    peak_ndx_adjusted = peak_ndx[0] + min_ndx
    bpm = 60.0 / peak_ndx_adjusted * (fs / max_decimation)
    print(bpm)
    return bpm, correl

def detect_midi(mk):
    midiout = rtmidi.MidiOut()
    for port, name in enumerate(midiout.get_ports()):
        if re.search(mk,name):
            midiOn = True
        else:
            midiOn = False
    del midiout
    return midiOn

def init_bpm(mk):
    rp = midi.SysEx(mk)
    if mk == 'RP8000':
        # mk1 only - switch the display to bpm
        rp.tempo_mode()
    return rp

def set_bpm(bpm,rp):
    rp.set_tempo(bpm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .wav file to determine the Beats Per Minute.")
    parser.add_argument("--filename", required=True, help=".wav file for processing")
    parser.add_argument(
        "--window",
        type=float,
        default=3,
        help="Size of the the window (seconds) that will be scanned to determine the bpm. Typically less than 10 seconds. [3]",
    )

    args = parser.parse_args()
    mk = 'RP8000mk2'
    midiOn = False
    while not midiOn:
        midiOn = detect_midi(mk)
        time.sleep(60)

    rp = init_bpm(mk)
    while midiOn:   
        record_wav(args.filename,args.window + 1)
        samps, fs = read_wav(args.filename)
        data = []
        correl = []
        bpm = 0
        n = 0
        nsamps = len(samps)
        window_samps = int(args.window * fs)
        samps_ndx = 0  # First sample in window_ndx
        max_window_ndx = math.floor(nsamps / window_samps)
        bpms = numpy.zeros(max_window_ndx)

        # Iterate through all windows
        for window_ndx in range(0, max_window_ndx):

            # Get a new set of samples
            # print(n,":",len(bpms),":",max_window_ndx_int,":",fs,":",nsamps,":",samps_ndx)
            data = samps[samps_ndx : samps_ndx + window_samps]
            if not ((len(data) % window_samps) == 0):
                raise AssertionError(str(len(data)))

            bpm, correl_temp = bpm_detector(data, fs)
            if bpm is None:
                continue
            bpms[window_ndx] = bpm
            correl = correl_temp

            # Iterate at the end of the loop
            samps_ndx = samps_ndx + window_samps

            # Counter for debug...
            n = n + 1

        bpm = numpy.median(bpms)
        set_bpm(bpm,rp)
        print("Completed!  Estimated Beats Per Minute:", bpm)

        n = range(0, len(correl))
        plt.plot(n, abs(correl))
        plt.show(block=True)
        midiOn = detect_midi(mk)
