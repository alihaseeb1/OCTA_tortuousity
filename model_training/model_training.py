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
data_dir = ""

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

train_paths, val_paths, train_labels, val_labels = train_test_split(all_image_paths, all_labels, test_size=0.10, random_state = 1)

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

class CustomDataGenerator(Sequence):
    def __init__(self, paths, labels, batch_size = 32, shuffle = True, augment = False):
        """
        :param paths: List of image file paths
        :param labels: Corresponding list of integer labels (0 or 1)
        :param batch_size: Number of samples per batch
        :param shuffle: Whether to shuffle indices after each epoch
        :param augment: Whether to apply random augmentations
        """
        self.paths = paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # get batch index
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Collect batch data
        batch_paths = [self.paths[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]

        # Load and process data
        X, y = self.__load_batch(batch_paths, batch_labels, image_channels=IMG_CHANNELS)
        return X, y

    def on_epoch_end(self):
        """ Shuffle indexes after each epoch """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __load_batch(self, batch_paths, batch_labels, image_channels = 1):
        """
        Load, process and returns a batch of images and corresponding labels
        """
        X = np.zeros((len(batch_paths), IMG_HEIGHT, IMG_WIDTH, image_channels))
        y = np.array(batch_labels)

        for i, (path, label) in enumerate(zip(batch_paths, batch_labels)):
            img_np = load_and_pad_image(path, target_size=IMG_HEIGHT, to_RGB=image_channels == 3) # 224 and convert it to 3 channels for Densenet
            if self.augment:
                img_np = random_augment(img_np)
            X[i] = img_np
            y[i] = label
        return X, y
    

def resample_data(paths, labels, sampling_strategy = "auto"):
  # ros = RandomOverSampler(sampling_strategy=sampling_strategy)
  rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
  
  # let's just oversample for now
  resampled_paths, resampled_labels = rus.fit_resample(np.array(paths).reshape(-1, 1), labels)
  return resampled_paths.flatten(), resampled_labels

train_paths, train_labels = resample_data(train_paths, train_labels)

print(f"Total negative instances:{len([i for i in train_labels if i == 0])}, and total training examples: {len(train_labels)}")

train_gen = CustomDataGenerator(train_paths, train_labels, batch_size=BATCH_SIZE, shuffle=True, augment=True)
val_gen = CustomDataGenerator(val_paths, val_labels, batch_size=BATCH_SIZE, shuffle=False, augment=False)
# test_gen = CustomDataGenerator(test_paths, test_labels, batch_size=BATCH_SIZE, shuffle=False, augment=False)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_check = ModelCheckpoint('model_tortuous.keras', monitor='val_loss', save_best_only=True)

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


densenet_base = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling = "avg")

for layer in densenet_base.layers:
    layer.trainable = False 

model = models.Sequential([
    densenet_base,
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation = "sigmoid")
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


EPOCHS = 100

# history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)

# change class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}
# class_weights = {0: 1., 1: 7.}  # Adjust the weight for the minority class

# history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, class_weight=class_weights, callbacks=[early_stopping, model_check])
history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=[early_stopping, model_check])


X_val = []
for i, (path, label) in enumerate(zip(val_paths, val_labels)):
    img_np = load_and_pad_image(path, target_size=IMG_HEIGHT, to_RGB=True) # 224 and convert it to 3 channels for Densenet
    X_val.append(img_np)

X_val = np.array(X_val)  # Shape should now be (num_samples, 224, 224, 3)

print(X_val.shape)
# Get predictions from the model
y_pred_prob = model.predict(X_val, batch_size=32, verbose=1)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary labels

# Calculate classification metrics
report = classification_report(val_labels, y_pred, target_names=['Non-Tortuous', 'Tortuous'])
print(report)

# test_loss, test_acc = model.evaluate(test_gen)
# print(f"Test accuracy: {test_acc}")

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, label='Training loss')
plt.plot(epochs, val_loss_values, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')

model.save('final_model_tortuous.keras')