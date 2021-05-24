# NS_DTLN
双lstm的语音降噪模型

# Requirements
tensorflow 2+
train_env.yml中包含了训练需要的python包

# train
1.准备好数据集test和train，每个数据集需要有clear音频和noisy音频，对应的音频命名一样。
2. 运行run_training.py

# eval
准备好噪音文件夹和训练好的模型。运行run_evaluation.py。
以下链接为已训练好的模型：
https://download.csdn.net/download/qq_41854731/19038499
