from model_training.model_config import load_and_pad_image, focal_loss, f1_score
import tensorflow as tf
import os
import glob
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
from .visualize_img import get_visualized_img
from .calculate_dice import calculate_dice_score
from .config import data_dirs, original_img_files, PROB_THRESHOLD
import pandas as pd
from tqdm import tqdm

# Load the model
try:
    model = tf.keras.models.load_model(
            'model_eval/model.keras',
            custom_objects={
                'loss': focal_loss,
                'f1_m': f1_score
                }  # Add your custom loss function here
        )
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

accuracies, precisions, recalls, f1_scores, dice_indexes = [], [], [], [], []

BATCH_SIZE = 32

weights_for_image = [] # is needed for the average dice score calculation (we store the number of pixels in the ground truth)

# Process each dataset

# Initialize the dataframe to store results
metrics_df = pd.DataFrame(columns=[
    'Directory', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Dice Score', 'Weighted Dice Score'
])

confusion_matrices = []  # List to store confusion matrices for each dataset
dice_scores = []  # Store dice scores
dirs_for_conf = [] # directory for name of confusion matrix file

# Process each dataset
for i, data_dir in enumerate(data_dirs):
    # Gather image paths and labels
    tortuous_paths = glob.glob(os.path.join(data_dir, "tortuous", "*"))
    non_tortuous_paths = glob.glob(os.path.join(data_dir, "non_tortuous", "*"))
    all_image_paths = tortuous_paths + non_tortuous_paths
    all_labels = [1] * len(tortuous_paths) + [0] * len(non_tortuous_paths)

    # Initialize variables for metrics
    tp, tn, fp, fn = 0, 0, 0, 0
    pred_prob, pred_labels, predicted_tortuous_paths = [], [], []

    # Initialize lists for batching
    batch_images = []
    batch_labels = []
    batch_image_path = []


    # Process images in batches
    for idx, image_path in tqdm(enumerate(all_image_paths), total=len(all_image_paths), desc="Processing and predicting images"):
        try:
            img = load_and_pad_image(image_path, 224, True)
            batch_images.append(img)
            label = all_labels[idx]
            batch_labels.append(label)
            batch_image_path.append(image_path)
            # Once we reach the batch size, make predictions
            if len(batch_images) == BATCH_SIZE or idx == len(all_image_paths) - 1:
                # Convert list of images to numpy array
                batch_images_np = np.array(batch_images)
                batch_labels_np = np.array(batch_labels)

                # Make predictions for the batch
                preds = model(batch_images_np, training=False).numpy()

                # Process each prediction
                for j, (pred, true_label) in enumerate(zip(preds, batch_labels_np)):
                    pred_prob.append(pred)
                    pred_label = 1 if pred > PROB_THRESHOLD else 0
                    pred_labels.append(pred_label)

                    if pred_label == 1:
                        # print(batch_image_path[j])
                        predicted_tortuous_paths.append(batch_image_path[j])

                    # Update confusion matrix counters
                    if pred_label == 1 and true_label == 1:
                        tp += 1
                    elif pred_label == 0 and true_label == 0:
                        tn += 1
                    elif pred_label == 1 and true_label == 0:
                        fp += 1
                    elif pred_label == 0 and true_label == 1:
                        fn += 1

                # Reset batch lists
                batch_images = []
                batch_labels = []
                batch_image_path = []

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

    # Compute metrics
    accuracy = accuracy_score(all_labels, pred_labels)
    precision = precision_score(all_labels, pred_labels)
    recall = recall_score(all_labels, pred_labels)
    f1 = f1_score(all_labels, pred_labels)

    report = classification_report(all_labels, pred_labels, target_names=['Non-Tortuous', 'Tortuous'])
    print(report)
    # Store the results in the dataframe
    metrics_df = pd.concat([metrics_df, pd.DataFrame([{
        'Directory': data_dir,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Dice Score': None,  # Dice Score to be calculated later
        'Weighted Dice Score': None  # Weighted Dice Score to be calculated later
    }])], ignore_index=True)

    # Print results
    print("=" * 50)
    print(f"For directory: {data_dir}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Print confusion matrix
    conf_matrix = confusion_matrix(all_labels, pred_labels)
    conf_matrix = pd.DataFrame(conf_matrix, index=["Actual Non_tortuous", "Actual Tortuous"], columns=["Predicted Non_tortuous", "Predicted Tortuous"])
    print("Confusion Matrix:")
    print(conf_matrix)

    dirs_for_conf.append(data_dir)
    confusion_matrices.append(conf_matrix)

    # Visualize results
    ground = []
    weight = 0
    predicted = []

    if tortuous_paths:
        ground, weight = get_visualized_img(original_img_files[i], tortuous_paths, is_prediction=False)
    if predicted_tortuous_paths:
        predicted, _ = get_visualized_img(original_img_files[i], predicted_tortuous_paths, is_prediction=True)

    weights_for_image.append(weight)

    dice_index = calculate_dice_score(ground, predicted)
    dice_scores.append(dice_index)

    # Update the dataframe with dice score
    metrics_df.at[i, 'Dice Score'] = dice_index

    print(f"Dice Score is: {dice_index}")
    print("=" * 50)

# Final statistics
print("Final Prediction Stats:")
if len(metrics_df) > 0:
    print(f"Accuracy: {metrics_df['Accuracy'].mean():.4f}")
    print(f"Precision: {metrics_df['Precision'].mean():.4f}")
    print(f"Recall: {metrics_df['Recall'].mean():.4f}")
    print(f"F1 Score: {metrics_df['F1 Score'].mean():.4f}")

print("=" * 50)

# Compute and save weighted dice score
weights_sum = sum(weights_for_image)
if weights_sum == 0:
    weighted_dice_score = 0
else:
    weighted_dice_score = np.sum(np.array(weights_for_image) * np.array(dice_scores)) / sum(weights_for_image) 

metrics_df['Weighted Dice Score'] = weighted_dice_score

print(f"Weighted Final Dice Score: {weighted_dice_score}")

# Save metrics to CSV
metrics_df.to_csv('model_eval/final_csvs/final_results.csv', index=False)

# Save confusion matrices to CSV (optional)
for idx, conf_matrix in enumerate(confusion_matrices):
    print(dirs_for_conf[idx])
    # assumes that the path is
    # model_eval/images\{file_name}\result
    conf_matrix.to_csv(f"model_eval/final_csvs/confusion_matrix_{idx}.csv", index=True)
    
print("=" * 50)