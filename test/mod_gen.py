
import os
import uuid
import time
import numpy as np
import tensorflow as tf
import external.luodong.Model2bin as mem_to_bin

mod_uuid = uuid.uuid4()
TODAY = time.strftime("%Y%m%d")
model_ids = list(np.arange(22))
mem_path = r'/home/hangz/TF_graphs/unit_test/DTLN_keras/converted_model/'
os.makedirs(os.path.join(mem_path, 'upload'), exist_ok=True)
mem_to_bin.auto_create_bin(mem_path=mem_path, model_ids=model_ids, file_prefix='dtln', mod_uuid=mod_uuid, today=TODAY)

