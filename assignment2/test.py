import getpass
import sys
import time

import numpy as np
from copy import deepcopy

import tensorflow as tf
from tensorflow.python.ops.seq2seq import sequence_loss

with tf.Session() as session:
  embed_size=4
  step_size=3
  i = tf.constant([1, 2, 3, 4,5,0],   shape=[2, step_size])
  L = tf.get_variable("L", shape=(6, embed_size))
  embeddings = tf.nn.embedding_lookup(L, tf.transpose(i))
  tensor = embeddings
  
  init = tf.initialize_all_variables()
  session.run(init) 
  
  print L.eval()

  output = tensor.eval()
  print output.shape
  print output

  output = tf.split(0, step_size, tensor)
  for x in output:
    print tf.squeeze(x).eval().shape
    print tf.squeeze(x).eval()
