import sys
import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Activation, Multiply

time_steps = 2

block_len = 128
block_shift = 64
num_units = 64
encoder_size = 128

wav_path = r'../data/noisy/noisy_snr_1.wav'
model_path = r'/home/hangz/TF_graphs/unit_test/DTLN_keras/models_DTLN_noFFT_saved_model/'

np.set_printoptions(threshold=sys.maxsize, precision=4, suppress=True)

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
lstm_1_kernel = model.variables[5].numpy()
lstm_1_recurrent_kernel = model.variables[6].numpy()
lstm_1_bias = model.variables[7].numpy()
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
lstm = LSTM(num_units, return_sequences=True, return_state=True, stateful=False, activation='tanh', recurrent_activation='sigmoid', use_bias=True)
output_lstm, _, cell_lstm = lstm(output_dense_i)
lstm.set_weights([lstm_kernel, lstm_recurrent_kernel, lstm_bias])
# lstm_1
lstm_1 = LSTM(num_units, return_sequences=True, return_state=True, stateful=False, activation='tanh', recurrent_activation='sigmoid', use_bias=True)
output_lstm_1, _, cell_lstm_1 = lstm_1(output_lstm)
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


for i in range(time_steps):
    in_buffer = audio[block_shift*i:block_len+block_shift*i]
    in_block = np.expand_dims(in_buffer, axis=0).astype(np.float32)

    # framing
    output_frame = tf.signal.frame(in_block, block_len, block_shift)
    # dense_i
    output_dense_i = dense_i(output_frame)
    # lstm
    output_lstm, _, cell_lstm = lstm(output_dense_i)
    # lstm_1
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


from tensorflow import keras


xt = output_dense_i[:, 0, :]
ft_n = np.matmul(xt, lstm_kernel[:, : num_units]) + np.matmul(output_lstm, lstm_recurrent_kernel[:, : num_units])
ft = keras.activations.sigmoid(ft_n + lstm_bias[: num_units])

with open('test.txt', 'w') as f:
    f.write('x\n')
    f.write(str(xt) + '\n')
    f.write('npu_out\n')
    f.write(str(ft_n) + '\n')
    f.write('add biase\n')
    f.write(str(ft_n + lstm_bias[: num_units]) + '\n')
    f.write('out\n')
    f.write(str(ft) + '\n')


# it = keras.activations.sigmoid(np.matmul(xt, kernel_i) + np.matmul(hidden_t_, recurrent_kernel_i) + bias_i)
# ct = keras.activations.tanh(np.matmul(xt, kernel_c) + np.matmul(hidden_t_, recurrent_kernel_c) + bias_c)
# ot = keras.activations.sigmoid(np.matmul(xt, kernel_o) + np.matmul(hidden_t_, recurrent_kernel_o) + bias_o)
# cell_t = cell_t_ * ft + it * ct
# hidden_t = ot * keras.activations.tanh(cell_t)

