import sys
import numpy as np
import soundfile as sf
import tensorflow as tf

time_steps = 10

block_len = 128
block_shift = 64

wav_path = r'../data/noisy/noisy_snr_1.wav'
model_path = r'../models_DTLN_noFFT_saved_model/'

np.set_printoptions(threshold=sys.maxsize, precision=6, suppress=True)

model = tf.saved_model.load(model_path)
infer = model.signatures["serving_default"]

audio, fs = sf.read(wav_path)

with open('./input_output_{:}.txt'.format(time_steps), 'w') as f:
    for i in range(time_steps):
        in_buffer = audio[block_shift*i:block_len+block_shift*i]
        in_block = np.expand_dims(in_buffer, axis=0).astype(np.float32)
        out_block = infer(tf.constant(in_block))['dense_o']
        out_buffer = np.squeeze(out_block)
        f.write('[{:}] input\n'.format(i+1))
        f.write(str(in_buffer) + '\n')
        f.write('[{:}] output\n'.format(i+1))
        f.write(str(out_buffer) + '\n')
        print(np.sum(np.abs(out_buffer)))

