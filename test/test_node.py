import sys
import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Activation, Multiply

time_steps = 100

block_len = 128
block_shift = 64
num_units = 64
encoder_size = 128

wav_path = r'../data/noisy/noisy_snr_1.wav'
model_path = r'/home/hangz/TF_graphs/unit_test/DTLN_keras/models_DTLN_noFFT_saved_model/'

np.set_printoptions(threshold=sys.maxsize, precision=6, suppress=True)

audio, fs = sf.read(wav_path)

model = tf.saved_model.load(model_path)
# dense_i/kernel:0
# dense_i/bias:0
# lstm/lstm_cell/kernel:0
# lstm/lstm_cell/recurrent_kernel:0
# lstm/lstm_cell/bias:0
# lstm_1/lstm_cell_1/kernel:0
# lstm_1/lstm_cell_1/recurrent_kernel:0
# lstm_1/lstm_cell_1/bias:0
# dense/kernel:0
# dense/bias:0
# dense_o/kernel:0
dense_i_kernel = model.variables[0].numpy()
dense_i_bias = model.variables[1].numpy()
lstm_kernel = model.variables[2].numpy()
lstm_recurrent_kernel = model.variables[3].numpy()
lstm_bias = model.variables[4].numpy()
lstm_kernel_i = lstm_kernel[:, : num_units]
lstm_kernel_f = lstm_kernel[:, num_units : num_units*2]
lstm_kernel_c = lstm_kernel[:, num_units*2 : num_units*3]
lstm_kernel_o = lstm_kernel[:, num_units*3 :]
lstm_recurrent_kernel_i = lstm_recurrent_kernel[:, : num_units]
lstm_recurrent_kernel_f = lstm_recurrent_kernel[:, num_units : num_units*2]
lstm_recurrent_kernel_c = lstm_recurrent_kernel[:, num_units*2 : num_units*3]
lstm_recurrent_kernel_o = lstm_recurrent_kernel[:, num_units*3 :]
lstm_bias_i = lstm_bias[: num_units]
lstm_bias_f = lstm_bias[num_units : num_units*2]
lstm_bias_c = lstm_bias[num_units*2 : num_units*3]
lstm_bias_o = lstm_bias[num_units*3 :]
lstm_1_kernel = model.variables[5].numpy()
lstm_1_recurrent_kernel = model.variables[6].numpy()
lstm_1_bias = model.variables[7].numpy()
lstm_1_kernel_i = lstm_1_kernel[:, : num_units]
lstm_1_kernel_f = lstm_1_kernel[:, num_units : num_units*2]
lstm_1_kernel_c = lstm_1_kernel[:, num_units*2 : num_units*3]
lstm_1_kernel_o = lstm_1_kernel[:, num_units*3 :]
lstm_1_recurrent_kernel_i = lstm_1_recurrent_kernel[:, : num_units]
lstm_1_recurrent_kernel_f = lstm_1_recurrent_kernel[:, num_units : num_units*2]
lstm_1_recurrent_kernel_c = lstm_1_recurrent_kernel[:, num_units*2 : num_units*3]
lstm_1_recurrent_kernel_o = lstm_1_recurrent_kernel[:, num_units*3 :]
lstm_1_bias_i = lstm_1_bias[: num_units]
lstm_1_bias_f = lstm_1_bias[num_units : num_units*2]
lstm_1_bias_c = lstm_1_bias[num_units*2 : num_units*3]
lstm_1_bias_o = lstm_1_bias[num_units*3 :]
dense_kernel = model.variables[8].numpy()
dense_bias = model.variables[9].numpy()
dense_o_kernel = model.variables[10].numpy()

in_buffer = np.zeros((block_len))
in_block = np.expand_dims(in_buffer, axis=0).astype(np.float32)
# framing
output_frame = tf.signal.frame(in_block, block_len, block_shift)
# dense_i
dense_i = Conv1D(encoder_size, 1, strides=1, use_bias=True)
output_dense_i = dense_i(output_frame)
dense_i.set_weights([dense_i_kernel, dense_i_bias])
# lstm
lstm = LSTM(num_units, return_sequences=True, return_state=True, stateful=True, activation='tanh', recurrent_activation='sigmoid', use_bias=True)
output_lstm, _, cell_lstm = lstm(output_dense_i * 0)
lstm.set_weights([lstm_kernel, lstm_recurrent_kernel, lstm_bias])
# lstm_1
lstm_1 = LSTM(num_units, return_sequences=True, return_state=True, stateful=True, activation='tanh', recurrent_activation='sigmoid', use_bias=True)
output_lstm_1, _, cell_lstm_1 = lstm_1(output_lstm * 0)
lstm_1.set_weights([lstm_1_kernel, lstm_1_recurrent_kernel, lstm_1_bias])
# dense
dense = Dense(encoder_size, use_bias=True)
output_dense = dense(output_lstm_1)
dense.set_weights([dense_kernel, dense_bias])
# sigmoid
output_sigmoid = Activation('sigmoid')(output_dense)
# multiply
output_multiply = Multiply()([output_dense_i, output_sigmoid])
# dense_o
dense_o = Conv1D(block_len, 1, padding='causal', use_bias=False)
output_dense_o = dense_o(output_multiply)
dense_o.set_weights([dense_o_kernel])
# squeeze
out_buffer = np.squeeze(output_dense_o)

lstm_hidden = output_lstm * 0
lstm_cell = cell_lstm * 0
lstm_1_hidden = output_lstm_1 * 0
lstm_1_cell = cell_lstm_1 * 0
with open('./node_{:}.txt'.format(time_steps), 'w') as f:
    for i in range(time_steps):
        in_buffer = audio[block_shift*i:block_len+block_shift*i]
        in_block = np.expand_dims(in_buffer, axis=0).astype(np.float32)
        # framing
        output_frame = tf.signal.frame(in_block, block_len, block_shift)
        # dense_i
        output_dense_i = dense_i(output_frame)
        # lstm
        lstm_ft = keras.activations.sigmoid(np.matmul(output_dense_i, lstm_kernel_f) + np.matmul(lstm_hidden, lstm_recurrent_kernel_f) + lstm_bias_f)
        lstm_it = keras.activations.sigmoid(np.matmul(output_dense_i, lstm_kernel_i) + np.matmul(lstm_hidden, lstm_recurrent_kernel_i) + lstm_bias_i)
        lstm_ct = keras.activations.tanh(np.matmul(output_dense_i, lstm_kernel_c) + np.matmul(lstm_hidden, lstm_recurrent_kernel_c) + lstm_bias_c)
        lstm_ot = keras.activations.sigmoid(np.matmul(output_dense_i, lstm_kernel_o) + np.matmul(lstm_hidden, lstm_recurrent_kernel_o) + lstm_bias_o)
        lstm_cell = lstm_cell * lstm_ft + lstm_it * lstm_ct
        lstm_hidden = lstm_ot * keras.activations.tanh(lstm_cell)
        output_lstm, _, cell_lstm = lstm(output_dense_i)
        # lstm_1
        lstm_1_ft = keras.activations.sigmoid(np.matmul(output_lstm, lstm_1_kernel_f) + np.matmul(lstm_1_hidden, lstm_1_recurrent_kernel_f) + lstm_1_bias_f)
        lstm_1_it = keras.activations.sigmoid(np.matmul(output_lstm, lstm_1_kernel_i) + np.matmul(lstm_1_hidden, lstm_1_recurrent_kernel_i) + lstm_1_bias_i)
        lstm_1_ct = keras.activations.tanh(np.matmul(output_lstm, lstm_1_kernel_c) + np.matmul(lstm_1_hidden, lstm_1_recurrent_kernel_c) + lstm_1_bias_c)
        lstm_1_ot = keras.activations.sigmoid(np.matmul(output_lstm, lstm_1_kernel_o) + np.matmul(lstm_1_hidden, lstm_1_recurrent_kernel_o) + lstm_1_bias_o)
        lstm_1_cell = lstm_1_cell * lstm_1_ft + lstm_1_it * lstm_1_ct
        lstm_1_hidden = lstm_1_ot * keras.activations.tanh(lstm_1_cell)
        output_lstm_1, _, cell_lstm_1 = lstm_1(output_lstm)
        # dense
        output_dense = dense(output_lstm_1)
        # sigmoid
        output_sigmoid = Activation('sigmoid')(output_dense)
        # multiply
        output_multiply = Multiply()([output_dense_i, output_sigmoid])
        # dense_o
        output_dense_o = dense_o(output_multiply)
        # squeeze
        out_buffer = np.squeeze(output_dense_o)

        f.write('[{:}] in_buffer\n'.format(i+1))
        f.write(str(in_buffer) + '\n')
        f.write('[{:}] output_frame\n'.format(i+1))
        f.write(str(output_frame) + '\n')
        f.write('[{:}] output_dense_i\n'.format(i+1))
        f.write(str(output_dense_i) + '\n')
        # f.write('[{:}] lstm_cell\n'.format(i+1))
        # f.write(str(lstm_cell) + '\n')
        # f.write('[{:}] lstm_hidden\n'.format(i+1))
        # f.write(str(lstm_hidden) + '\n')
        f.write('[{:}] cell_lstm\n'.format(i+1))
        f.write(str(cell_lstm) + '\n')
        f.write('[{:}] output_lstm\n'.format(i+1))
        f.write(str(output_lstm) + '\n')
        # f.write('[{:}] lstm_1_cell\n'.format(i+1))
        # f.write(str(lstm_1_cell) + '\n')
        # f.write('[{:}] lstm_1_hidden\n'.format(i+1))
        # f.write(str(lstm_1_hidden) + '\n')
        f.write('[{:}] cell_lstm_1\n'.format(i+1))
        f.write(str(cell_lstm_1) + '\n')
        f.write('[{:}] output_lstm_1\n'.format(i+1))
        f.write(str(output_lstm_1) + '\n')
        f.write('[{:}] output_dense\n'.format(i+1))
        f.write(str(output_dense) + '\n')
        f.write('[{:}] output_sigmoid\n'.format(i+1))
        f.write(str(output_sigmoid) + '\n')
        f.write('[{:}] output_multiply\n'.format(i+1))
        f.write(str(output_multiply) + '\n')
        f.write('[{:}] output_dense_o\n'.format(i+1))
        f.write(str(output_dense_o) + '\n')
        f.write('[{:}] out_buffer\n'.format(i+1))
        f.write(str(out_buffer) + '\n')
