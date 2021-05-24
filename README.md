# NS_DTLN
双lstm语音降噪模型

# Requirements
tensorflow 2+
train_env.yml中包含训练需要的python包

# train
1.准备好音频数据集test和train，每个数据集需要包含clean和noisy数据集，其中对应的音频文件命名应该一样
2.运行run_training.py

# eval
准备好需要降噪的音频文件夹和训练好的.h5格式模型，运行run_evaluation.py
以下是训练好的降噪模型
https://download.csdn.net/download/qq_41854731/19038499

