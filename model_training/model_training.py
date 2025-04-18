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

from model_config import IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH, load_and_pad_image, random_augment, train_labels, train_paths, val_labels, val_paths, BATCH_SIZE, focal_loss, f1_score


# alpha and gamma for focal loss

ALPHA = 0.9
GAMMA = 3.0



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

# train_paths, train_labels = resample_data(train_paths, train_labels)

print(f"Total negative instances:{len([i for i in train_labels if i == 0])}, and total training examples: {len(train_labels)}")

train_gen = CustomDataGenerator(train_paths, train_labels, batch_size=BATCH_SIZE, shuffle=True, augment=True)
val_gen = CustomDataGenerator(val_paths, val_labels, batch_size=BATCH_SIZE, shuffle=False, augment=False)
# test_gen = CustomDataGenerator(test_paths, test_labels, batch_size=BATCH_SIZE, shuffle=False, augment=False)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_check = ModelCheckpoint(f'best_model_dense_{ALPHA}_{GAMMA}.keras', monitor='val_loss', save_best_only=True)


densenet_base = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling = "avg")

for layer in densenet_base.layers[:-5]:
    layer.trainable = False 

model = models.Sequential([
    densenet_base,
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation = "sigmoid")
])


model.compile(optimizer='adam', loss=focal_loss(alpha=ALPHA, gamma=GAMMA), metrics=['accuracy'])

model.summary()


EPOCHS = 1

# history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)

# change class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
# class_weights = {i: class_weights[i] for i in range(len(class_weights))}
# class_weights = {0: 1., 1: 7.}  # Adjust the weight for the minority class

# history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, class_weight=class_weights, callbacks=[early_stopping, model_check])
history = model.fit(train_gen, epochs=EPOCHS,validation_data=val_gen, callbacks=[early_stopping, model_check])

model.save(f'final_model_dense_{ALPHA}_{GAMMA}.keras')


X_val = []
for i, (path, label) in enumerate(zip(val_paths, val_labels)):
    img_np = load_and_pad_image(path, target_size=IMG_HEIGHT, to_RGB=True) # 224 and convert it to 3 channels for Densenet
    X_val.append(img_np)

X_val = np.array(X_val)  # Shape should now be (num_samples, 224, 224, 3)

print(X_val.shape)
# Get predictions from the model
y_pred_prob = model.predict(X_val, batch_size=32, verbose=1)
y_pred = (y_pred_prob > 0.6).astype(int)  # Convert probabilities to binary labels

# Calculate classification metrics
print("=" * 30)
print("Final Results on the Validation Set:")
print("=" * 30)
report = classification_report(val_labels, y_pred, target_names=['Non-Tortuous', 'Tortuous'])
print(report)

misclassified_indices = []

for i, pred in enumerate(y_pred):
    if pred[0] != val_labels[i]:
        misclassified_indices.append(i) 

# Show misclassified images
# for idx in misclassified_indices[:5]:  # Show first 5 misclassified images
#     # Load the image from the path
#     img_np = load_and_pad_image(val_paths[idx], target_size=IMG_HEIGHT, to_RGB=True)
#     true_label = val_labels[idx]
#     pred_label = y_pred[idx][0]

#     # Display the misclassified image with its true and predicted labels
#     plt.imshow(img_np)
#     plt.title(f"True: {'Tortuous' if true_label == 1 else 'Non-Tortuous'}, Pred: {'Tortuous' if pred_label == 1 else 'Non-Tortuous'}")
#     plt.axis('off')  # Turn off axes for a cleaner view
#     plt.show()