import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import numpy as np
import cv2
from IPython.display import Image as DisplayImage   

input_names = ['image_tensor']
IMAGE_PATH = "./data/dogs.jpg"

# The TensorRT inference graph file downloaded from Colab or your local machine.
pb_fname = "./trt_graph.pb"

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


trt_graph = get_frozen_graph(pb_fname)