#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from mpi4py import MPI
from pathlib import Path, PurePath
from time import time, sleep
import multiprocessing
import os
import argparse
import gc
import scipy
import signal
from scipy.signal import butter, lfilter, freqz, correlate2d
import glob
import essentia
import essentia.standard as es
import essentia.streaming as ess
from essentia.standard import *
import time as time
import datetime
import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '/home/bqpd/Desktop/MusicSimilarity-UI/rp_extract')
from audiofile_read import * # reading wav and mp3 files
from rp_feature_io import CSVFeatureWriter, HDF5FeatureWriter, read_csv_features, load_multiple_hdf5_feature_files
import rp_extract as rp # Rhythm Pattern extractor
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5 import uic
import sys

fs = 44100
octave = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

np.set_printoptions(threshold=np.inf)

gc.enable()
filelist = []
for filename in Path('music').glob('**/*.mp3'):
    filelist.append(filename)
for filename in Path('music').glob('**/*.wav'):
    filelist.append(filename)  
print("length of filelist" + str(len(filelist)))

# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object

do_mfcc_kl = 1
do_mfcc_euclid = 1
do_notes = 1
do_chroma = 1
do_bh = 1
startbatch = 0
endbatch = 1000000
batchsize = 25
#TestApp().run()

# Python 3 compatibility hack
try:
    unicode('')
except NameError:
    unicode = str

form_class = uic.loadUiType("ms.ui")[0]  # Load the UI

class MyWindowClass(QMainWindow, form_class):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)

    def extrbutton_clicked(self):
        options = QFileDialog.Options()
        fileName = QFileDialog.getExistingDirectory(self,"Select folder containing music", "", options=options)
        if fileName:
            print("Selected Folder" + fileName)

            filelist = []
            for filename in Path(str(fileName)).glob('**/*.mp3'):
                filelist.append(filename)
            for filename in Path(str(fileName)).glob('**/*.wav'):
                filelist.append(filename)  

            print("length of filelist" + str(len(filelist)))

            extract_all_rhythm_feats(fileName)
            print("Extracting Rhythm Features") 
            extract_all_rhythm_feats(fileName)

            print("Extracting Melodic Features") 
            time_dict = {}
            tic1 = int(round(time.time() * 1000))
            # BATCH FEATURE EXTRACTION:
            process_stuff(startbatch, endbatch, batchsize, do_mfcc_kl, do_mfcc_euclid, do_notes, do_chroma, do_bh)
            tac1 = int(round(time.time() * 1000))
            time_dict['MPI TIME FEATURE']= tac1 - tic1
            #if rank == 0:
            print("Process " + str(rank) + " time: " + str(time_dict)) 

    def loadbutton_clicked(self):
        options = QFileDialog.Options()
        fileName = QFileDialog.getExistingDirectory(self,"Select folder containing feature files", "","All Files (*)", options=options)
        if fileName:
            print(fileName)

    def selectbutton_clicked(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,"Select a song", "","All Files (*)", options=options)
        if fileName:
            print(fileName)

def parallel_python_process(process_id, cpu_filelist, f_mfcc_kl, f_mfcc_euclid, f_notes, f_chroma, f_bh):
    #return (end_time - start_time)
    #PARAMETER: mfcc_kl, mfcc_euclid, notes, chroma, bh
    if f_mfcc_euclid == 1:    
        with open("features1/out" + str(process_id) + ".mfcc", "w") as myfile:
            myfile.write("")
            myfile.close()
    if f_mfcc_kl == 1:    
        with open("features1/out" + str(process_id) + ".mfcckl", "w") as myfile:
            myfile.write("")
            myfile.close()
    if f_chroma == 1:
        with open("features1/out" + str(process_id) + ".chroma", "w") as myfile:
            myfile.write("")
            myfile.close()       
    if f_bh == 1:   
        with open("features1/out" + str(process_id) + ".bh", "w") as myfile:
            myfile.write("")
            myfile.close()
    if f_notes == 1:        
        with open("features1/out" + str(process_id) + ".notes", "w") as myfile:
            myfile.write("")
            myfile.close()
    count = 1
    for file_name in cpu_filelist:
        path = str(PurePath(file_name))
        print ("File " + path + " " + str(count) + " von " + str(len(cpu_filelist))) 
        bpmret, hist, key, scale, notes, chroma_matrix, mean, cov, var, cov_kl = compute_features(path, f_mfcc_kl, f_mfcc_euclid, f_notes, f_chroma, f_bh)
        if key == 0:
            continue
        filename = path.replace(".","").replace(";","").replace(",","").replace("mp3",".mp3").replace("aiff",".aiff").replace("aif",".aif").replace("au",".au").replace("m4a", ".m4a").replace("wav",".wav").replace("flac",".flac").replace("ogg",".ogg")  # rel. filename as from find_files
        if f_mfcc_euclid == 1:                
            with open("features1/out" + str(process_id) + ".mfcc", "a") as myfile:
                print ("MFCC File " + path + " " + str(count) + " von " + str(len(cpu_filelist))) 
                str_mean = np.array2string(mean, precision=8, separator=',', suppress_small=True).replace('\n', '')#.strip('[ ]')
                str_var = np.array2string(var, precision=8, separator=',', suppress_small=True).replace('\n', '')#.strip('[ ]')
                str_cov = np.array2string(cov, precision=8, separator=',', suppress_small=True).replace('\n', '')#.strip('[ ]')
                line = (filename + "; " + str_mean + "; " + str_var + "; " + str_cov).replace('\n', '')
                myfile.write(line + '\n')       
                myfile.close()
        if f_chroma == 1:  
            with open("features1/out" + str(process_id) + ".chroma", "a") as myfile:
                print ("Chroma Full - File " + path + " " + str(count) + " von " + str(len(cpu_filelist))) 
                transposed_chroma = np.zeros(chroma_matrix.shape)
                transposed_chroma = transpose_chroma_matrix(key, scale, chroma_matrix)
                chroma_str = np.array2string(transposed_chroma.transpose(), separator=',', suppress_small=True).replace('\n', '')
                line = (filename + "; " + chroma_str).replace('\n', '')
                myfile.write(line + '\n')       
                myfile.close()
        if f_bh == 1:  
            with open("features1/out" + str(process_id) + ".bh", "a") as myfile:
                print ("Beat Histogram - File " + path + " " + str(count) + " von " + str(len(cpu_filelist))) 
                bpmret = str(bpmret)
                hist = np.array2string(hist, separator=',', suppress_small=True).replace('\n', '')
                line = (filename + "; " + bpmret + "; " + hist).replace('\n', '')
                myfile.write(line + '\n')       
                myfile.close()
        if f_notes == 1:  
            with open("features1/out" + str(process_id) + ".notes", "a") as myfile:
                print ("Chroma Notes - File " + path + " " + str(count) + " von " + str(len(cpu_filelist))) 
                key = str(key)
                transposed_notes = []
                transposed_notes = transpose_chroma_notes(key, scale, notes)
                #print notes
                scale = str(scale).replace('\n', '')
                transposed_notes = str(transposed_notes).replace('\n', '')
                line = (filename + "; " + key + "; " + scale + "; " + transposed_notes).replace('\n', '')
                myfile.write(line + '\n')       
                myfile.close()
        if f_mfcc_kl == 1:                
            with open("features1/out" + str(process_id) + ".mfcckl", "a") as myfile:
                print ("MFCC Kullback-Leibler " + path + " " + str(count) + " von " + str(len(cpu_filelist))) 
                str_mean = np.array2string(mean, precision=8, separator=',', suppress_small=True).replace('\n', '')#.strip('[ ]')
                str_cov_kl = np.array2string(cov_kl, precision=8, separator=',', suppress_small=True).replace('\n', '')#.strip('[ ]')
                line = (filename + "; " + str_mean + "; " + str_cov_kl).replace('\n', '')
                myfile.write(line + '\n')       
                myfile.close()
        count = count + 1
        del bpmret, hist, key, scale, notes, chroma_matrix, mean, cov, var, cov_kl
        gc.enable()
        gc.collect()
    gc.enable()
    gc.collect()
    return 1

def parallel_python_process_files(process_id, cpu_filelist):
    print("calling rank " + str(rank) + " size " + str(size))
    count = 1
    for file_name in cpu_filelist:
        path = str(PurePath(file_name))
        filename = path.replace(".","").replace(";","").replace(",","").replace("mp3",".mp3").replace("aiff",".aiff").replace("aif",".aif").replace("au",".au").replace("m4a", ".m4a").replace("wav",".wav").replace("flac",".flac").replace("ogg",".ogg")  # rel. filename as from find_files
        with open("features1/out" + str(process_id) + ".files", "a") as myfile:
            #print ("File " + path + " " + str(count) + " von " + str(len(cpu_filelist))) 
            line = (filename + "     :       " + str(process_id))
            myfile.write(line + '\n')       
            myfile.close()
        count = count + 1
        gc.enable()
        gc.collect()
    gc.enable()
    gc.collect()
    return 1

def process_stuff(startjob, maxparts, batchsz, f_mfcc_kl, f_mfcc_euclid, f_notes, f_chroma, f_bh):
    startjob = int(startjob)
    maxparts = int(maxparts) + 1
    files_per_part = int(batchsz)
    print("starting with: ")    
    print(startjob)
    print("ending with: ")
    print(maxparts - 1)
    # Divide the task into subtasks - such that each subtask processes around 25 songs
    print("files per part: ")
    print(files_per_part)
    start = 0
    end = len(filelist)
    print("used cores: " + str(size))
    ncpus = size
    parts = int(round(len(filelist) / files_per_part) + 1)
    print("Split problem in parts: ")
    print(str(parts))
    step = int((end - start) / parts + 1)
    if maxparts > parts:
        maxparts = parts
    for index in range(startjob + rank, maxparts, size):
        if index < parts:        
            starti = start+index*step
            endi = min(start+(index+1)*step, end)
            print("calling process  " + str(rank) + " index " + str(index) + " size " + str(size) + " starti " + str(starti) + " endi " + str(endi))
            parallel_python_process(index, filelist[int(starti):int(endi)], f_mfcc_kl, f_mfcc_euclid, f_notes, f_chroma, f_bh)
            with open("features1/out" + str(index) + ".files", "w") as myfile:
                myfile.write("")
                myfile.close()
            parallel_python_process_files(index, filelist[starti:endi])
            gc.collect()
    gc.enable()
    gc.collect()

def transpose_chroma_matrix(key, scale, chroma_param):
    if key == 'Ab':
        key = 'G#'
    if key == 'Gb':
        key = 'F#'
    if key == 'Eb':
        key = 'D#'
    if key == 'Db':
        key = 'C#'
    if key == 'Bb':
        key = 'A#'
    chroma_param = chroma_param.transpose()
    transposed_chroma = np.zeros(chroma_param.shape)
    if key != 'A':
        #print("transposing: ")
        #get key offset
        offs = 12 - octave.index(key)
        #print(offs)
        for ind in range(len(chroma_param)):
            #print "original" + str(ind)
            index = (ind + offs)
            if(index >= 12):
                index = index - 12
            #print "new" + str(index)
            transposed_chroma[index] = chroma_param[ind]
    else:
        transposed_chroma = chroma_param
    transposed_chroma = transposed_chroma.transpose()
    #print transposed_chroma[0:4]
    return transposed_chroma

def transpose_chroma_notes(key, scale, notes):
    if key == 'Ab':
        key = 'G#'
    if key == 'Gb':
        key = 'F#'
    if key == 'Eb':
        key = 'D#'
    if key == 'Db':
        key = 'C#'
    if key == 'Bb':
        key = 'A#'
    transposed = notes
    if key != 'A':
        #print("transposing: ")
        #get key offset
        offs = 12 - octave.index(key)
        #print(offs)
        index = 0
        for i in notes:
            i = i + offs
            if(i >= 12):
                i = i - 12
            transposed[index] = i
            index = index + 1
    return transposed

def compute_features(path, f_mfcc_kl, f_mfcc_euclid, f_notes, f_chroma, f_bh):
    gc.enable()        
    # Loading audio file
    #will resample if sampleRate is different!
    try: 
        audio = es.MonoLoader(filename=path, sampleRate=fs)()
    except: 
        print("Erroneos File detected by essentia standard: skipping!")
        #return bpm, histogram, key, scale, notes, chroma_matrix, mean, cov, var, cov_kl  
        return 0, [], 0, 0, [], [], [], [], [], []
    #will resample if sampleRate is different!
    try: 
        loader = ess.MonoLoader(filename=path, sampleRate=44100)
    except:
        print("Erroneos File detected by essentia streaming: skipping!")
        #return bpm, histogram, key, scale, notes, chroma_matrix, mean, cov, var, cov_kl  
        return 0, [], 0, 0, [], [], [], [], [], []
    #Initialize algorithms we will use
    frameSize = 4096#512
    hopSize = 2048#256
    #######################################
    # DO FILTERING ONLY FOR MFCC - not with essentia standard
    # below is just an example
    #HP = es.HighPass(cutoffFrequency=128)
    #LP = es.LowPass(cutoffFrequency=4096)
    #lp_f = LP(audio)
    #hp_f = HP(lp_f)
    #audio = hp_f
    #MonoWriter(filename='music/filtered.wav')(filtered_audio)
    HP = ess.HighPass(cutoffFrequency=128)
    LP = ess.LowPass(cutoffFrequency=4096)
    #loader = ess.MonoLoader(filename=path, sampleRate=44100)
    #writer = ess.MonoWriter(filename='music/filtered.wav')
    #frameCutter = FrameCutter(frameSize = 1024, hopSize = 512)
    #pool = essentia.Pool()
    # Connect streaming algorithms
    #loader.audio >> HP.signal
    #HP.signal >> LP.signal
    #LP.signal >> writer.audio
    # Run streaming network
    #essentia.run(loader)
    bpm = 0
    histogram = 0
    key = 0
    scale = 0
    notes = 0
    chroma_matrix = 0
    mean = 0  
    cov = 0
    var = 0
    cov_kl = 0
    #####################################
    # extract mfcc
    #####################################
    if f_mfcc_kl == 1 or f_mfcc_euclid == 1:
        #features, features_frames = es.MusicExtractor(analysisSampleRate=44100, mfccStats=['mean', 'cov'])(path)
        #m, n = features['lowlevel.mfcc.cov'].shape
        #print m
        #iu1 = np.triu_indices(m)
        #cov = features['lowlevel.mfcc.cov'][iu1]
        #mean = features['lowlevel.mfcc.mean']
        #print(features['lowlevel.mfcc.cov'])
        hamming_window = es.Windowing(type='hamming')
        spectrum = es.Spectrum()  # we just want the magnitude spectrum
        mfcc = es.MFCC(numberCoefficients=13)
        frame_sz = 2048#512
        hop_sz = 1024#256
        mfccs = np.array([mfcc(spectrum(hamming_window(frame)))[1]
                       for frame in es.FrameGenerator(audio, frameSize=frame_sz, hopSize=hop_sz)])
        #Let's scale the MFCCs such that each coefficient dimension has zero mean and unit variance:
        #mfccs = sklearn.preprocessing.scale(mfccs)
        #print mfccs.shape
        mean = np.mean(mfccs.T, axis=1)
        #print(mean)
        var = np.var(mfccs.T, axis=1)
        #print(var)
        cov = np.cov(mfccs.T)
        cov_kl = cov#.flatten()
        #get only upper triangular matrix values to shorten length
        iu1 = np.triu_indices(13)
        cov = cov[iu1]
        #plt.imshow(mfccs.T, origin='lower', aspect='auto', interpolation='nearest')
        #plt.ylabel('MFCC Coefficient Index')
        #plt.xlabel('Frame Index')    
        #plt.colorbar()
    #####################################
    # extract beat features and histogram
    #####################################
    if f_bh == 1 or f_chroma == 1 or f_notes == 1: 
        # Compute beat positions and BPM
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
        if f_bh == 1:
            peak1_bpm, peak1_weight, peak1_spread, peak2_bpm, peak2_weight, peak2_spread, histogram = es.BpmHistogramDescriptors()(beats_intervals)
        tempo = bpm
        times = beats
        beats_frames = (beats * fs) / hopSize
        beats_frames = beats_frames.astype(int)

        #fig, ax = plt.subplots()
        #ax.bar(range(len(histogram)), histogram, width=1)
        #ax.set_xlabel('BPM')
        #ax.set_ylabel('Frequency')
        #plt.title("BPM histogram")
        #ax.set_xticks([20 * x + 0.5 for x in range(int(len(histogram) / 20))])
        #ax.set_xticklabels([str(20 * x) for x in range(int(len(histogram) / 20))])
        #plt.show()

    #####################################
    # extract full beat aligned chroma
    #####################################

    framecutter = ess.FrameCutter(frameSize=frameSize, hopSize=hopSize, silentFrames='noise')
    windowing = ess.Windowing(type='blackmanharris62')
    spectrum = ess.Spectrum()
    spectralpeaks = ess.SpectralPeaks(orderBy='magnitude',
                                      magnitudeThreshold=0.00001,
                                      minFrequency=20,
                                      maxFrequency=3500,
                                      maxPeaks=60)
    # Use default HPCP parameters for plots, however we will need higher resolution
    # and custom parameters for better Key estimation
    hpcp = ess.HPCP()
    hpcp_key = ess.HPCP(size=36, # we will need higher resolution for Key estimation
                        referenceFrequency=440, # assume tuning frequency is 44100.
                        bandPreset=False,
                        minFrequency=20,
                        maxFrequency=3500,
                        weightType='cosine',
                        nonLinear=False,
                        windowSize=1.)
    key = ess.Key(profileType='edma', # Use profile for electronic music
                  numHarmonics=4,
                  pcpSize=36,
                  slope=0.6,
                  usePolyphony=True,
                  useThreeChords=True)
    # Use pool to store data
    pool = essentia.Pool()
    # Connect streaming algorithms
    ###################################
    # USE FILTER - comment next lines in
    loader.audio >> HP.signal
    HP.signal >> LP.signal
    LP.signal >> framecutter.signal
    ###################################
    ###################################
    # NO FILTER - comment next line in
    #loader.audio >> framecutter.signal        
    ###################################
    framecutter.frame >> windowing.frame >> spectrum.frame
    spectrum.spectrum >> spectralpeaks.spectrum
    spectralpeaks.magnitudes >> hpcp.magnitudes
    spectralpeaks.frequencies >> hpcp.frequencies
    spectralpeaks.magnitudes >> hpcp_key.magnitudes
    spectralpeaks.frequencies >> hpcp_key.frequencies
    hpcp_key.hpcp >> key.pcp
    hpcp.hpcp >> (pool, 'tonal.hpcp')
    key.key >> (pool, 'tonal.key_key')
    key.scale >> (pool, 'tonal.key_scale')
    key.strength >> (pool, 'tonal.key_strength')
    # Run streaming network
    essentia.run(loader)
    #print("Estimated key and scale:", pool['tonal.key_key'] + " " + pool['tonal.key_scale'])
    #print(pool['tonal.hpcp'].T)
    chroma = pool['tonal.hpcp'].T
    key = pool['tonal.key_key'] 
    scale = pool['tonal.key_scale']
    if f_chroma == 1:
        # Plot HPCP
        #imshow(pool['tonal.hpcp'].T, aspect='auto', origin='lower', interpolation='none')
        #plt.title("HPCPs in frames (the 0-th HPCP coefficient corresponds to A)")
        #show()
        #print beats_frames.shape[0]
        chroma_matrix = np.zeros((beats_frames.shape[0], 12))
        prev_beat = 0
        act_beat = 0
        sum_key = np.zeros(12)
        chroma_align = chroma
        chroma_align = chroma_align.transpose()
        mat_index = 0
        for i in beats_frames:
            act_beat = i
            value = sum(chroma_align[prev_beat:act_beat])/(act_beat-prev_beat)
            chroma_align[prev_beat:act_beat] = value
            prev_beat = i
            if np.linalg.norm(value, ord=1) != 0:
                value = value / np.linalg.norm(value, ord=1)
            chroma_matrix[mat_index] = value
            mat_index = mat_index + 1

        #chroma_align = chroma_align.transpose()   
        #plt.figure(figsize=(10, 4))
        #librosa.display.specshow(chroma_align, y_axis='chroma', x_axis='time')
        #plt.vlines(times, 0, 12, alpha=0.5, color='r', linestyle='--', label='Beats')
        #plt.colorbar()
        #plt.title('Chromagram')
        #plt.tight_layout()
        #chroma_align = chroma_align.transpose()
    #print(chroma_align[24:28])
    #####################################
    # extract full chroma text
    #####################################
    if f_notes == 1:
        #print(chroma.shape)
        m, n = chroma.shape
        avg = 0
        chroma = chroma.transpose()
        m, n = chroma.shape
        for j in chroma:
            avg = avg + np.sum(j)
        avg = avg / m
        threshold = avg / 2
        for i in chroma:
            if np.sum(i) > threshold:
                ind = np.where(i == np.max(i))
                max_val = i[ind]#is always 1!
                i[ind] = 0

                ind2 = np.where(i == np.max(i))
                i[ind] = 1

                #if np.any(i[ind2][0] >= 0.8 * max_val):
                    #i[ind2] = i[ind2]
                    #pass
                #low_values_flags = i < 1
                low_values_flags = i < 0.8

                i[low_values_flags] = 0
            else:
                i.fill(0)     
        chroma = chroma.transpose()
        # Compute beat positions and BPM
        prev_beat = 0
        act_beat = 0
        sum_key = np.zeros(12)
        chroma = chroma.transpose()  
        for i in beats_frames:
            act_beat = i
            sum_key = sum(chroma[prev_beat:act_beat])
            #print(sum_key)
            #print(chroma[prev_beat:act_beat])

            ind = np.where(sum_key == np.max(sum_key))
            ind = ind[0]
            #print("debug")
            fill = np.zeros(len(j))
            if(np.all(chroma[prev_beat:act_beat] == 0)):
                fill[ind] = 0
            else:    
                fill[ind] = 1
            chroma[prev_beat:act_beat] = fill
            #print(chroma[prev_beat:act_beat])
            prev_beat = i
            #print("BEAT")
        notes = []
        for i in notes:
            del i
        prev_beat = 0
        act_beat = 0    
        for i in beats_frames:
            act_beat = i
            sum_key = sum(chroma[prev_beat:act_beat])
            ind = np.where(sum_key == np.max(sum_key))
            prev_beat = i
            notes.append(ind[0][0])
            prev_beat = i 
        #chroma = chroma.transpose()  
        #plt.figure(figsize=(10, 4))
        #librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
        #plt.vlines(times, 0, 12, alpha=0.5, color='r', linestyle='--', label='Beats')
        #plt.colorbar()
        #plt.title('Chromagram')
        #plt.tight_layout()
        #chroma = chroma.transpose() 
    gc.collect()
    return bpm, histogram, key, scale, notes, chroma_matrix, mean, cov, var, cov_kl

'''
RP_extract: Rhythm Patterns Audio Feature Extractor
@author: 2014-2015 Alexander Schindler, Thomas Lidy
'''
def read_feature_files(filenamestub,ext,separate_ids=True,id_column=0):
    from rp_feature_io import read_csv_features
    return read_csv_features(filenamestub,ext,separate_ids,id_column)

'''
RP_extract: Rhythm Patterns Audio Feature Extractor
@author: 2014-2015 Alexander Schindler, Thomas Lidy
'''
def timestr(seconds):
    ''' returns HH:MM:ss formatted time string for given seconds
    (seconds can be a float with milliseconds included, but only the integer part will be used)
    :return: string
    '''
    if seconds is None:
        return "--:--:--"
    else:
        return str(datetime.timedelta(seconds=int(seconds)))

'''
RP_extract: Rhythm Patterns Audio Feature Extractor
@author: 2014-2015 Alexander Schindler, Thomas Lidy
'''
def find_files(path,file_types=('.wav','.mp3'),relative_path = False,verbose=False,ignore_hidden=True):
    ''' function to find all files of a particular file type in a given path
    path: input path to start searching
    file_types: a tuple of file extensions (e.g.'.wav','.mp3') (case-insensitive) or 'None' in which case ALL files in path will be returned
    relative_path: if False, absolute paths will be returned, otherwise the path relative to the given path
    verbose: will print info about files found in path if True
    ignore_hidden: if True (default) will ignore Linux hidden files (starting with '.')
    '''
    if path.endswith(os.sep):
        path = path[0:-1]   # we need to remove the file separator at the end otherwise the path handling below gets confused
    # lower case the file types for comparison
    if file_types: # if we have file_types (otherwise 'None')
        if type(file_types) == tuple:
            file_types = tuple((f.lower() for f in file_types))
            file_type_string = ' or '.join(file_types) # for print message only
        else: # single string
            file_types = file_types.lower()
            file_type_string = file_types # for print message only
    else:
        file_type_string = 'any file type'  # for print message only
    all_files = []
    for d in os.walk(unicode(path)):    # finds all subdirectories and gets a list of files therein
        # subpath: complete sub directory path (full path)
        # filelist: files in that sub path (filenames only)
        (subpath, _, filelist) = d
        if ignore_hidden:
            filelist = [ file for file in filelist if not file[0] == '.']
        if file_types:   # FILTER FILE LIST by FILE TYPE
            filelist = [ file for file in filelist if file.lower().endswith(file_types) ]
        if (verbose): print(subpath,":", len(filelist), "files found (" + file_type_string + ")")
        # add full absolute path
        filelist = [ subpath + os.sep + file for file in filelist ]
        if relative_path: # cut away full path at the beginning (+/- 1 character depending if path ends with path separator)
            filelist = [ filename[len(path)+1:] for filename in filelist ]
        all_files.extend(filelist)
    return all_files


# mp3_to_wav_batch:
# finds all MP3s in a given directory in all subdirectories
# and converts all of them to WAV
# if outdir is specified it will replicate the entire subdir structure from within input path to outdir
# otherwise the WAV file will be created in the same dir as the MP3 file
# in both cases the file name is maintained and the extension changed to .wav
# Example for MP3 to WAV batch conversion (in a new Python script):
# from rp_extract_batch import mp3_to_wav_batch
# mp3_to_wav_batch('/data/music/ISMIRgenre/mp3_44khz_128kbit_stereo','/data/music/ISMIRgenre/wav')

'''
RP_extract: Rhythm Patterns Audio Feature Extractor
@author: 2014-2015 Alexander Schindler, Thomas Lidy
'''
def mp3_to_wav_batch(path,outdir=None,audiofile_types=('.mp3','.aif','.aiff')):
    get_relative_path = (outdir!=None) # if outdir is specified we need relative path otherwise absolute
    filenames = find_files(path,audiofile_types,get_relative_path)
    n_files = len(filenames)
    n = 0
    for file in filenames:
        n += 1
        basename, ext = os.path.splitext(file)
        wav_file = basename + '.wav'
        if outdir: # if outdir is specified we add it in front of the relative file path
            file = path + os.sep + file
            wav_file = outdir + os.sep + wav_file
            # recreate same subdir path structure as in input path
            out_subpath = os.path.split(wav_file)[0]
            if not os.path.exists(out_subpath):
                os.makedirs(out_subpath)
        # future option: (to avoid recreating the input path subdir structure in outdir)
        #filename_only = os.path.split(wav_file)[1]
        try:
            if not os.path.exists(wav_file):
                print("Decoding:", n, "/", n_files, ":")
                if ext.lower() == '.mp3':
                    mp3_decode(file,wav_file)
                elif ext.lower() == '.aif' or ext.lower() == '.aiff':
                    cmd = ['ffmpeg','-v','1','-y','-i', file,  wav_file]
                    return_code = subprocess.call(cmd)  # subprocess.call takes a list of command + arguments
                    if return_code != 0:
                        raise DecoderException("Problem appeared during decoding.", command=cmd)
            else:
                print("Already existing: " + wav_file)
        except:
            print("Not decoded " + file)

'''
RP_extract: Rhythm Patterns Audio Feature Extractor
@author: 2014-2015 Alexander Schindler, Thomas Lidy
'''
def extract_all_files_in_path(in_path,
                              out_file = None,
                              feature_types = ['rp','ssd','rh'],
                              audiofile_types=('.wav','.mp3'),
                              label=False,
                              verbose=True):
    """
    finds all files of a certain type (e.g. .wav and/or .mp3) in a path and all sub-directories in it
    extracts selected RP feature types
    and saves them into separate CSV feature files (one per feature type)
    # path: input file path to search for audio files (including subdirectories)
    # out_file: output file name stub for feature files to write (if omitted, features will be returned from function)
    # feature_types: RP feature types to extract. see rp_extract.py
    # audiofile_types: a string or tuple of suffixes to look for file extensions to consider (include the .)
    # label: use subdirectory name as class label
    """
    # get file list of all files in a path (filtered by audiofile_types)
    filelist = find_files(in_path,audiofile_types,relative_path=True)
    return extract_all_files(filelist, in_path, out_file, feature_types, label, verbose)

'''
RP_extract: Rhythm Patterns Audio Feature Extractor
@author: 2014-2015 Alexander Schindler, Thomas Lidy
'''
def extract_all_files_generic(in_path,
                              out_file = None,
                              feature_types = ['rp','ssd','rh'],
                              audiofile_types=('.wav','.mp3'),
                              path_prefix=None,
                              label=False,
                              append=False,
                              append_diff=False,
                              no_extension_check=False,
                              force_resampling=None,
                              out_HDF5 = False,
                              log_AudioTypes = True,
                              log_Errors = True,
                              verbose=True):
    """
    finds all files of a certain type (e.g. .wav and/or .mp3) in a path (+ sub-directories)
    OR loads a list of files to extract from a given .txt file
    extracts selected RP feature types
    and saves them into separate CSV feature files (one per feature type)
    # in_path: input file path to search for audio files (including subdirectories) OR .txt file containing a list of filenames
    # out_file: output file name stub for feature files to write (if omitted, features will be returned from function)
    # feature_types: RP feature types to extract. see rp_extract.py
    # audiofile_types: a string or tuple of suffixes to look for file extensions to consider (include the .)
    # path_prefix: prefix to be added to relative filenames (used typically together with .txt input files)
    # label:
    # append: append features to existing feature files
    # append_diff: append new features to existing output file(s) only if they are not in it/them yet
    # no_extension_check: does not check file format via extension. means that decoder is called on ALL files.
    # force_resampling: force a target sampling rate (provided in Hz) when decoding (works with FFMPEG only!)
    # out_HDF5: whether to store as HDF5 file format (otherwise CSV)
    # log_AudioTypes: creates a log file with audio format info
    # log_Errors: creates an error log file collecting all errors that appeared during feature extraction
    # verbose: verbose output or not
    """
    if in_path.lower().endswith('.txt'):  # treat as input file list
        from classes_io import read_filenames
        filelist = read_filenames(in_path)
        in_path = path_prefix # in case path_prefix is passed it is added to files in extract_all_files
    elif os.path.isdir(in_path): # find files in path
        if no_extension_check: audiofile_types = None # override filetypes to include all files (no extension check)
        filelist = find_files(in_path,audiofile_types,relative_path=True)
        # filelist will be relative, so we provide in_path below
    elif in_path.lower().endswith(audiofile_types) or no_extension_check: # treat as single audio input file
        filelist = [in_path]
        in_path = None # no abs path to add below
    else:
        raise ValueError("Cannot not process this kind of input file: " + in_path)
    if append_diff:
        # get differential filelist to extract only new feature files
        filelist = get_diff_filelist(out_file, filelist, feature_types, out_HDF5)
        append = True
    startjob = int(startbatch)
    maxparts = int(endbatch) + 1
    files_per_part = int(batchsize)
    print("starting with: ")    
    print(startjob)
    print("ending with: ")
    print(maxparts - 1)
    # Divide the task into subtasks - such that each subtask processes around 25 songs
    print("files per part: ")
    print(files_per_part)
    start = 0
    end = len(filelist)
    print("used cores: " + str(size))
    ncpus = size
    parts = int(round(len(filelist) / files_per_part) + 1)
    print("Split problem in parts: ")
    print(str(parts))
    step = (end - start) / parts + 1
    if maxparts > parts:
        maxparts = parts
    for index in range(int(startjob + rank), int(maxparts), int(size)):
        if index < parts:        
            starti = int(start+index*step)
            endi = int(min(start+(index+1)*step, end))
            print("calling process  " + str(rank) + " index " + str(index) + " size " + str(size) + " starti " + str(starti) + " endi " + str(endi))
            extract_all_files(filelist[starti:endi], in_path, out_file + str(index), feature_types, label, append, no_extension_check, force_resampling, out_HDF5, log_AudioTypes, log_Errors, verbose)
            gc.collect()
    gc.enable()
    gc.collect()
    return 

'''
RP_extract: Rhythm Patterns Audio Feature Extractor
@author: 2014-2015 Alexander Schindler, Thomas Lidy
'''
def extract_all_files(filelist,
                      path,
                      out_file=None,
                      feature_types =['rp','ssd','rh'],
                      label=False,
                      append=False,
                      no_extension_check=False,
                      force_resampling=None,
                      out_HDF5=False,
                      log_AudioTypes=True,
                      log_Errors=True,
                      verbose=True):
    """
    finds all files of a certain type (e.g. .wav and/or .mp3) in a path and all sub-directories in it
    extracts selected RP feature types
    and saves them into separate CSV feature files (one per feature type)
    # filelist: list of files for features to be extracted
    # path: absolute path that will be added at beginning of filelist (can be '')
    # out_file: output file name stub for feature files to write (if omitted, features will be returned from function)
    # feature_types: RP feature types to extract. see rp_extract.py
    # label: use subdirectory name as class label
    # no_extension_check: does not check file format via extension. means that decoder is called on ALL files.
    # force_resampling: force a target sampling rate (provided in Hz) when decoding (works with FFMPEG only!)
    # out_HDF5: whether to store as HDF5 file format (otherwise CSV)
    """
    ext = feature_types
    n = 0   # counting the files being processed
    n_extracted = 0   # counting the files that were actually analyzed
    err = 0 # counting errors
    n_files = len(filelist)
    # initialize filelist_extracted and dict containing all accumulated feature arrays
    filelist_extracted = []
    feat_array = {}
    audio_logwriter = None
    error_logwriter = None
    audio_logwriter_wrote_header = False
    start_time = time.time()
    if out_file: # only if out_file is specified
        if log_AudioTypes:
            pass
            #log_filename = out_file + '.audiotypes.log'
            #audio_logfile = open(log_filename, 'w') # TODO allow append mode 'a'
            #audio_logwriter = unicsv.UnicodeCSVWriter(audio_logfile) #, quoting=csv.QUOTE_ALL)
        if log_Errors:
            pass            
            #err_log_filename = out_file + '.errors.log'
            #error_logfile = open(err_log_filename, 'w') # TODO allow append mode 'a'
            #error_logwriter = unicsv.UnicodeCSVWriter(error_logfile) #, quoting=csv.QUOTE_ALL)
        if out_HDF5:
            FeatureWriter = HDF5FeatureWriter()
        else:
            FeatureWriter = CSVFeatureWriter()
            FeatureWriter.open(out_file,ext,append=append)
    for fil in filelist:  # iterate over all files
        try:
            if n > 0:
                elaps_time = time.time() - start_time
                remain_time = elaps_time * n_files / n - elaps_time # n is the number of files done here
            else:
                remain_time = None
            n += 1
            if path:
                filename = path + os.sep + fil
            else:
                filename = fil
            if verbose:
                print('#',n,'/',n_files,'(ETA: ' + timestr(remain_time) + "):", filename)
            # read audio file (wav or mp3)
            samplerate, samplewidth, data, decoder = audiofile_read(filename, verbose=verbose, include_decoder=True, no_extension_check=no_extension_check, force_resampling=force_resampling)
            # audio file info
            if verbose: print(samplerate, "Hz,", data.shape[1], "channel(s),", data.shape[0], "samples")
            # extract features
            # Note: the True/False flags are determined by checking if a feature is listed in 'ext' (see settings above)
            feat = rp.rp_extract(data,
                              samplerate,
                              extract_rp   = ('rp' in ext),          # extract Rhythm Patterns features
                              extract_ssd  = ('ssd' in ext),           # extract Statistical Spectrum Descriptor
                              extract_tssd = ('tssd' in ext),          # extract temporal Statistical Spectrum Descriptor
                              extract_rh   = ('rh' in ext),           # extract Rhythm Histogram features
                              extract_trh  = ('trh' in ext),          # extract temporal Rhythm Histogram features
                              extract_mvd  = ('mvd' in ext),        # extract Modulation Frequency Variance Descriptor
                              spectral_masking=True,
                              transform_db=True,
                              transform_phon=True,
                              transform_sone=True,
                              fluctuation_strength_weighting=True,
                              skip_leadin_fadeout=1,
                              step_width=1,
                              verbose = verbose)
            # TODO check if ext and feat.keys are consistent
            # WHAT TO USE AS ID (based on filename): 3 choices:
            id = fil.replace(".","").replace(";","").replace(",","").replace("mp3",".mp3").replace("aiff",".aiff").replace("aif",".aif").replace("au",".au").replace("m4a", ".m4a").replace("wav",".wav").replace("flac",".flac").replace("ogg",".ogg")  # rel. filename as from find_files
            # id = filename   # full filename incl. full path
            # id = filename[len(path)+1:] # relative filename only (extracted from path)
            if out_file:
                # WRITE each feature set to a CSV or HDF5 file
                id2 = None
                if label:
                    id2 = id.replace("\\","/").split("/")[-2].strip()
                if out_HDF5 and n_extracted==0:
                    # for HDF5 we need to know the vector dimension
                    # thats why we cannot open the file earlier
                    FeatureWriter.open(out_file,ext,feat,append=append) # append not working for now but possibly in future
                FeatureWriter.write_features(id,feat,id2)
            else:
                # IN MEMORY: add the extracted features for 1 file to the array dict accumulating all files
                # TODO: only if we don't have out_file? maybe we want this as a general option
                if feat_array == {}: # for first file, initialize empty array with dimension of the feature set
                    for e in feat.keys():
                        feat_array[e] = np.empty((0,feat[e].shape[0]))
                # store features in array
                for e in feat.keys():
                    feat_array[e] = np.append(feat_array[e], feat[e].reshape(1,-1), axis = 0) # 1 for horizontal vector, -1 means take original dimension
                filelist_extracted.append(id)
            n_extracted += 1
            # write list of analyzed audio files alongsize audio metadata (kHz, bit, etc.)
            if audio_logwriter:
                if not audio_logwriter_wrote_header: # write CSV header
                    log_info = ["filename","decoder","samplerate (kHz)","samplewidth (bit)","n channels","n samples"]
                    audio_logwriter.writerow(log_info)
                    audio_logwriter_wrote_header = True
                log_info = [filename,decoder,samplerate,samplewidth*8,data.shape[1],data.shape[0]]
                audio_logwriter.writerow(log_info)
            gc.collect() # after every file we do garbage collection, otherwise our memory is used up quickly for some reason
        except Exception as e:
            print("ERROR analysing file: " + fil + ": " + str(e))
            err += 1
            if error_logwriter:
                error_logwriter.writerow([fil,str(e)])
    try:
        if out_file:  # close all output files
            FeatureWriter.close()
            if audio_logwriter:
                audio_logfile.close()
        if error_logwriter:
            error_logfile.close()
    except Exception as e:
        print("ERROR closing the output or log files: " + str(e))
    end_time = time.time()
    if verbose:
        print("FEATURE EXTRACTION FINISHED.", n, "file(s) processed,", n_extracted, "successful. Duration:", timestr(end_time-start_time))
        if err > 0:
            print(err, "file(s) had ERRORs during feature extraction.",)
            if log_Errors:
                print ("See", err_log_filename)
            else:
                print()
        if out_file:
            opt_ext = '.h5' if out_HDF5 else ''
            print("Feature file(s):", out_file + "." + str(ext) + opt_ext)
    if out_file is None:
        return filelist_extracted, feat_array

def extract_all_rhythm_feats(param_folder):
    feature_types = []
    feature_types.append('rp')
    feature_types.append('rh')
    audiofile_types = get_supported_audio_formats()
    output_filename = "features1/out"
    input_path = str(param_folder)
    print("Extracting features:", feature_types)
    print("From files in:", input_path)
    print("File types:",)
    print(audiofile_types)
    time_dict = {}
    tic1 = int(round(time.time() * 1000))
    extract_all_files_generic(input_path,output_filename,feature_types, audiofile_types,
                              None, False, False, False, False, None,
                              False, log_AudioTypes = True)
    tac1 = int(round(time.time() * 1000))
    time_dict['MPI TIME FEATURE']= tac1 - tic1
    #if rank == 0:
    print("Process " + str(rank) + " time: " + str(time_dict)) 
    return 0

app = QApplication(sys.argv)
myWindow = MyWindowClass(None)
myWindow.show()
app.exec_()


