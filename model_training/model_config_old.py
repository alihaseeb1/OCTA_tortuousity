import os
import glob
import random
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import DenseNet121

from matplotlib import pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report

from tqdm import tqdm


BATCH_SIZE = 32
IMG_HEIGHT = 224 # after padding (we'll pad )
IMG_WIDTH = 224  # after padding
IMG_CHANNELS = 3 # we will increase if need be to match this
# data_dir = "dataset"
# data_dir = "C:\\Users\\aliha\\OCTA_tortuousity\\model_training\\Dataset"
data_dir = "C:\\Users\\aliha\\OCTA_tortuousity\\Dataset\\Dataset"

tortuous_paths = glob.glob(os.path.join(data_dir, "tortuous", "*"))
non_tortuous_paths = glob.glob(os.path.join(data_dir, "non_tortuous", "*"))

all_image_paths = []
all_labels = []
for p in tortuous_paths:
    all_image_paths.append(p)
    all_labels.append(1)
for p in non_tortuous_paths:
    all_image_paths.append(p)
    all_labels.append(0)


# train_paths, test_paths, train_labels, test_labels = train_test_split(all_image_paths, all_labels, test_size=0.2, random_state = 1)
# train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=0.25, random_state = 1)

train_paths, val_paths, train_labels, val_labels = train_test_split(all_image_paths, all_labels, test_size=0.15, random_state = 1)

print(f"Train: {len(train_paths)}")
print(f"Val: {len(val_paths)}")
# print(f"Test: {len(test_paths)}")


def load_and_pad_image(path, target_size = 1024, to_RGB = False):
    """
    - Opens the image in grayscale
    - If it's larger than target_size in any dimension, resizes down
    - Pads with black so final shape = target_size x target_size
    - Returns a float32 array in [0,1], shape (target_size, target_size, IMG_CHANNELS)
    """
    img = Image.open(path).convert("L")

    img_np = np.array(img, dtype=np.uint8)
    
    img_np = np.where(img_np >= 255, 255, 0).astype(np.uint8)

    img = Image.fromarray(img_np, mode="L")

    w , h = img.size

    ratio = min(target_size / w, target_size / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    new_img = Image.new("L", (target_size, target_size))
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    new_img.paste(img, (left, top))

    img_np = np.array(new_img, dtype=np.float32) / 255.0

    if to_RGB:
        img_np = np.stack((img_np,)*3, axis=-1)
    else:
        img_np = np.expand_dims(img_np, axis=-1)

    return img_np

def random_augment(img_np):
    if random.random() < 0.5:
        img_np = np.flip(img_np, axis=0)
    if random.random() < 0.5:
        img_np = np.flip(img_np, axis=1)
    k = random.randint(0, 3)
    img_np = np.rot90(img_np, k=k, axes=(0, 1))
    return img_np

@tf.keras.utils.register_keras_serializable()
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_weight = alpha_factor * K.pow((1 - pt), gamma)
        return K.mean(focal_weight * K.binary_crossentropy(y_true, y_pred))
    return loss


@tf.keras.utils.register_keras_serializable()
def f1_score(y_true, y_pred):
    # cast y_true to float32 so it matches y_pred's type
    y_true = K.cast(y_true, tf.float32)
    
    # threshold it at 0.5 
    y_pred = K.cast(K.greater(y_pred, 0.5), tf.float32)
    
    tp = K.sum(y_true * y_pred)
    tn = K.sum((1 - y_true) * (1 - y_pred))
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return f1
# img = load_and_pad_image("processed_normal (1)_non_tortuous_139.png", 224, to_RGB=True)
# plt.imshow(img, cmap='gray')
# plt.title('Skeletonized Image')
# plt.show()