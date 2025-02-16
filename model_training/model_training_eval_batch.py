import os
from sklearn.metrics import classification_report
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from model_config import val_paths, val_labels, IMG_CHANNELS, IMG_HEIGHT, load_and_pad_image, focal_loss, f1_score

# Directory containing the models
model_directory = "C:\\Users\\aliha\\OCTA_tortuousity\\model_training\\models"

# List all model files in the directory
model_files = [f for f in os.listdir(model_directory) if f.endswith('.keras')]

# Initialize a list to store the results
results = []

# Iterate over each model in the directory
for model_file in tqdm(model_files, unit="Models"):
    print(f"Processing model: {model_file}")
    
    # Load the model
    model_path = os.path.join(model_directory, model_file)
    model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'loss': focal_loss,
                'f1_m': f1_score
                }  # Add your custom loss function here
        )
    
    # Prepare validation data
    X_val = []
    for i, (path, label) in enumerate(zip(val_paths, val_labels)):
        img_np = load_and_pad_image(path, target_size=IMG_HEIGHT, to_RGB=True)  # 224 and convert it to 3 channels for Densenet
        X_val.append(img_np)

    X_val = np.array(X_val)  # Shape should now be (num_samples, 224, 224, 3)

    print(X_val.shape)

    # Get predictions from the model
    y_pred_prob = model.predict(X_val, batch_size=32, verbose=1)
    y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary labels

    # Calculate classification metrics
    report = classification_report(val_labels, y_pred, target_names=['Non-Tortuous', 'Tortuous'])
    print(report)

    # Find misclassified examples
    misclassified_indices = []
    
    for i, pred in enumerate(y_pred):
        if pred[0] != val_labels[i]:
            misclassified_indices.append(i) 
    
    # Show misclassified images
    for idx in misclassified_indices[:5]:  # Show first 5 misclassified images
        # Load the image from the path
        img_np = load_and_pad_image(val_paths[idx], target_size=IMG_HEIGHT, to_RGB=True)
        true_label = val_labels[idx]
        pred_label = y_pred[idx][0]

        # Display the misclassified image with its true and predicted labels
        plt.imshow(img_np)
        plt.title(f"True: {'Tortuous' if true_label == 1 else 'Non-Tortuous'}, Pred: {'Tortuous' if pred_label == 1 else 'Non-Tortuous'}")
        plt.axis('off')  # Turn off axes for a cleaner view
        plt.show()
    # Save model name and report into results
    report['model_name'] = model_file
    results.append(report)

# Convert the results into a DataFrame for easy viewing and saving
report_df = pd.DataFrame(results)

# Save the results to a CSV file
report_df.to_csv('model_classification_reports.csv', index=False)

print("Classification reports saved to 'model_classification_reports.csv'.")
