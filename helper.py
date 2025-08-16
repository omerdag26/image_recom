# helper.py
import tensorflow as tf
import os
def shut_gpu():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.config.set_visible_devices([], "GPU")  # comment to enable GPU



