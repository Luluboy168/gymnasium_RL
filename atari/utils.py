import cv2
import numpy as np
import tensorflow as tf

def preprocess_observation(obs):
    
    obs = tf.convert_to_tensor(obs)

    # Normalize
    if obs.dtype == tf.uint8:
        obs = tf.cast(obs, tf.float32) / 255.0
    else:
        obs = tf.clip_by_value(obs, 0.0, 1.0)

    # Convert to grayscale
    obs = tf.image.rgb_to_grayscale(obs)

    # Resize to 84x84 (auto-batched)
    obs = tf.image.resize(obs, [84, 84], method='area')

    return obs

