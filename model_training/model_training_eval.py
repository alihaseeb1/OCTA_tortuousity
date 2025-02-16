from sklearn.metrics import classification_report
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from model_info import val_paths, val_labels, IMG_CHANNELS, IMG_HEIGHT, load_and_pad_image

model = tf.keras.models.load_model('final_model_tortuous.keras')

X_val = []
for i, (path, label) in enumerate(zip(val_paths, val_labels)):
    img_np = load_and_pad_image(path, target_size=IMG_HEIGHT, to_RGB=True) # 224 and convert it to 3 channels for Densenet
    X_val.append(img_np)

X_val = np.array(X_val)  # Shape should now be (num_samples, 224, 224, 3)

print(X_val.shape)
# Get predictions from the model
y_pred_prob = model.predict(X_val, batch_size=32, verbose=1)
y_pred = (y_pred_prob > 0.7).astype(int)  # Convert probabilities to binary labels

# Calculate classification metrics
report = classification_report(val_labels, y_pred, target_names=['Non-Tortuous', 'Tortuous'])
print(report)


# # Find misclassified examples
# misclassified_indices = np.where(y_pred != val_labels)[0]

# # Show some of the misclassified images one by one
# import matplotlib.pyplot as plt

# for idx in misclassified_indices:  # Show first 5 misclassified images
#     # Load the image from disk again to avoid storing all in memory
#     img_np = load_and_pad_image(val_paths[idx], target_size=IMG_HEIGHT, to_RGB=True)
#     true_label = val_labels[idx]
#     pred_label = y_pred[idx]
    
#     # Display the misclassified image with its labels
#     plt.imshow(img_np)
#     plt.title(f"True: {true_label}, Pred: {pred_label}")
#     plt.show()