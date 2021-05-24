#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:23:15 2020

@author: nils
"""

import soundfile as sf
import numpy as np
import tensorflow as tf



##########################
# the values are fixed, if you need other values, you have to retrain.
# The sampling rate of 16k is also fix.
block_len = 256
block_shift = 64
# load model
model = tf.saved_model.load('../models_DTLN/models_DTLN_0731_01_saved_model')
infer = model.signatures["serving_default"]

snr_list = [0.1, 0.5, 1, 2, 5]
for snr in snr_list:
    # load audio file at 16k fs (please change)
    audio,fs = sf.read('./data/noisy/noisy_snr_'+str(snr)+'.wav')
    # check for sampling rate
    if fs != 8000:
        raise ValueError('This model only supports 8k sampling rate.')
    # preallocate output audio
    out_file = np.zeros((len(audio)))
    # create buffer
    in_buffer = np.zeros((block_len))
    out_buffer = np.zeros((block_len))
    # calculate number of blocks
    num_blocks = (audio.shape[0] - (block_len-block_shift)) // block_shift
    # iterate over the number of blcoks        
    for idx in range(num_blocks):
        # shift values and write to buffer
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        in_buffer[-block_shift:] = audio[idx*block_shift:(idx*block_shift)+block_shift]
        # create a batch dimension of one
        in_block = np.expand_dims(in_buffer, axis=0).astype('float32')
        # process one block
        out_block= infer(tf.constant(in_block))['conv1d_1']
        # shift values and write to buffer
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift))
        out_buffer  += np.squeeze(out_block)
        # write block to output file
        out_file[idx*block_shift:(idx*block_shift)+block_shift] = out_buffer[:block_shift]

    # write to .wav file 
    sf.write('../data/out/out_snr_'+str(snr)+'.wav', out_file, fs) 

print('Processing finished.')
