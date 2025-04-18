from ..model_training.model_config import load_and_pad_image, focal_loss, f1_score
import tensorflow as tf
import os
import glob
import cv2
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score as f1_sklearn, 
                             accuracy_score, confusion_matrix, classification_report)
from .visualize_img import get_visualized_img
from .calculate_dice import calculate_dice_score
from .config import original_img_files, PROB_THRESHOLDS
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------------------
# Load the model
# --------------------------------------------------------------------------------
try:
    model = tf.keras.models.load_model(
        './OCTA_tortuousity/model_eval/model.keras',
        custom_objects={'loss': focal_loss, 'f1_m': f1_score}
    )
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# --------------------------------------------------------------------------------
# We will store overall results for each threshold here, then pick the best threshold.
# --------------------------------------------------------------------------------
all_threshold_results = {}  # key = threshold, value = list of dataframes (one per directory)
# Also track the aggregated performance across *all* directories for each threshold
aggregate_results = []
data_dirs = ["./OCTA_tortuousity/Dataset/Dataset"]
# --------------------------------------------------------------------------------
# Main loop over directories
# --------------------------------------------------------------------------------
for i, data_dir in enumerate(data_dirs):
    # ------------------------------------------------------------
    # Gather all images and labels from this directory
    # ------------------------------------------------------------
    tortuous_paths = glob.glob(os.path.join(data_dir, "tortuous", "*"))
    non_tortuous_paths = glob.glob(os.path.join(data_dir, "non_tortuous", "*"))

    all_image_paths = tortuous_paths + non_tortuous_paths
    all_labels = [1]*len(tortuous_paths) + [0]*len(non_tortuous_paths)

    # ------------------------------------------------------------
    # Run the model predictions *once*; store probabilities
    # ------------------------------------------------------------
    BATCH_SIZE = 32
    batch_images = []
    batch_labels = []
    batch_image_paths = []

    pred_probs = []  # will store the raw probabilities for each image
    true_labels = []  # for cross-check
    all_img_paths = []

    for idx, image_path in tqdm(
        enumerate(all_image_paths),
        total=len(all_image_paths),
        desc=f"Gathering predictions for {data_dir}"
    ):
        try:
            img = load_and_pad_image(image_path, 224, True)
            batch_images.append(img)
            batch_labels.append(all_labels[idx])
            batch_image_paths.append(image_path)

            # If we reached the batch size or the last image, predict
            if len(batch_images) == BATCH_SIZE or idx == len(all_image_paths) - 1:
                batch_images_np = np.array(batch_images)
                preds = model(batch_images_np, training=False).numpy()

                # store these predictions
                pred_probs.extend(preds.flatten().tolist())  # flatten if shape=(N,1)
                true_labels.extend(batch_labels)
                all_img_paths.extend(batch_image_paths)

                # reset
                batch_images = []
                batch_labels = []
                batch_image_paths = []

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

    # ------------------------------------------------------------
    # For dice visualization, we need the tortuous and predicted paths
    # we only
    # re-run the final 'predicted_tortuous_paths' under the chosen threshold.
    # ------------------------------------------------------------
    # We'll gather the ground (from tortuous_paths) for dice calculation.
    # ground_img, weight = get_visualized_img(original_img_files[i], tortuous_paths, is_prediction=False)

    # ------------------------------------------------------------
    # Data structure to hold metrics for each threshold for this directory
    # We'll combine them later.
    # ------------------------------------------------------------
    directory_threshold_results = []

    # Evaluate across all thresholds in PROB_THRESHOLDS
    for threshold in PROB_THRESHOLDS:
        # For confusion matrix and standard metrics
        tp, tn, fp, fn = 0, 0, 0, 0

        # We'll need predicted_tortuous_paths for dice
        predicted_tortuous_paths = []

        # For applying threshold classification:
        for path, prob, label in zip(all_img_paths, pred_probs, true_labels):
            pred_label = 1 if prob > threshold else 0
            # for dice score calculation later on
            if pred_label == 1:
                predicted_tortuous_paths.append(path)

            if pred_label == 1 and label == 1:
                tp += 1
            elif pred_label == 0 and label == 0:
                tn += 1
            elif pred_label == 1 and label == 0:
                # wrongly predicted as tortuous
                fp += 1
            elif pred_label == 0 and label == 1:
                # wrongly predicted as non-tortuous
                fn += 1

        # Basic metrics
        all_pred_labels = [1 if p > threshold else 0 for p in pred_probs]
        accuracy = accuracy_score(true_labels, all_pred_labels)
        precision = precision_score(true_labels, all_pred_labels, zero_division=0)
        recall = recall_score(true_labels, all_pred_labels, zero_division=0)
        f1 = f1_sklearn(true_labels, all_pred_labels, zero_division=0)

        # confusion matrix
        conf_mat = confusion_matrix(true_labels, all_pred_labels)
        # classification_report for your reference
        report = classification_report(
            true_labels, all_pred_labels,
            target_names=['Non-Tortuous', 'Tortuous'],
            zero_division=0
        )

        print("=" * 50)
        print(f"For directory: {data_dir}, threshold={threshold}")
        print("=" * 50)
        print(report)
        print("Confusion Matrix:")
        print(pd.DataFrame(
            conf_mat,
            index=["Actual Non_tortuous", "Actual Tortuous"],
            columns=["Predicted Non_tortuous", "Predicted Tortuous"]
        ))

        # Dice Score
        # gather predicted overlay
        # predicted_img, _ = get_visualized_img(
        #     original_img_files[i],
        #     predicted_tortuous_paths,
        #     is_prediction=True
        # )
        # dice_index = calculate_dice_score(ground_img, predicted_img)
        # dice_weighted = dice_index * weight
        # Collect metrics in a dict
        metrics_dict = {
            'Directory': data_dir,
            'Threshold': threshold,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
            # 'Dice Score': dice_index,
            # "Weight * Dice": dice_weighted,
            # "Weight" : weight
        }

        directory_threshold_results.append(metrics_dict)
        # print(f"Dice Score is: {dice_index}")
        # print("=" * 50)

    # ------------------------------------------------------------
    # Make a DataFrame out of this directory’s threshold results
    # ------------------------------------------------------------
    directory_threshold_df = pd.DataFrame(directory_threshold_results)

    # ------------------------------------------------------------
    # Weighted Dice Score calculation for all thresholds
    # single `weight` we have for the entire directory.
    # We'll store the Weighted Dice Score in the same DataFrame:
    # if weight == 0:
    #     directory_threshold_df['Weighted Dice Score'] = 0
    # else:
    #     # same 'weight' for each threshold
    #     directory_threshold_df['Weighted Dice Score'] = (
    #         directory_threshold_df['Dice Score'] * weight
    #     )

    # ------------------------------------------------------------
    # Add this directory’s threshold-specific DF to a global structure
    # ------------------------------------------------------------
    all_threshold_results[data_dir] = directory_threshold_df

# --------------------------------------------------------------------------------
# Now we have a DataFrame per directory, each containing metrics for each threshold.
# We can pick the threshold that yields the highest F1 across ALL directories combined,
# or we can pick the best threshold per directory. 
# Below, I show how to pick the single best threshold globally by merging data first.
# --------------------------------------------------------------------------------

combined_df = pd.concat(all_threshold_results.values(), ignore_index=True)

# Group by threshold to get average metrics across all directories:
grouped_by_threshold = combined_df.groupby('Threshold').agg({
    'Accuracy': 'mean',
    'Precision': 'mean',
    'Recall': 'mean',
    'F1 Score': 'mean'
    # 'Dice Score': 'mean',
    # 'Weight': "sum",
    # "Weight * Dice" : sum
})
grouped_by_threshold.reset_index(inplace=True)

# Find the best threshold by max F1 Score
best_idx = grouped_by_threshold['F1 Score'].idxmax()
best_threshold = grouped_by_threshold.loc[best_idx, 'Threshold']

# grouped_by_threshold["Weighted Dice"] = (grouped_by_threshold["Weight * Dice"] / grouped_by_threshold["Weight"]).fillna(0)
 
print("\n========== Summary of thresholds (averaged across all directories) ==========")
print(grouped_by_threshold)
print(f"\nBest threshold based on F1 Score = {best_threshold}")

# Then save to CSV
best_threshold_df = combined_df[combined_df['Threshold'] == best_threshold].copy()
best_threshold_df.to_csv(
    './OCTA_tortuousity/model_eval/final_csvs/best_threshold_deep_training_optimized_metrics.csv',
    index=False
)



# # If you also want to save the per-directory confusion matrices for the best threshold:
# for i, data_dir in enumerate(data_dirs):
#     # Filter the rows for the best threshold
#     df_dir_best = all_threshold_results[data_dir]
#     df_dir_best = df_dir_best[df_dir_best['Threshold'] == best_threshold]

#     # You already have confusion matrices in the console output above.
#     # If you want to re-compute them and save to CSV here, do it similarly:
#     #   1) build confusion matrix from the predictions for best_threshold.
#     #   2) save it with a name that includes the directory or an index.

#     # For example:
#     # all_pred_labels = [1 if p > best_threshold else 0 for p in pred_probs]
#     # conf_matrix = confusion_matrix(true_labels, all_pred_labels)
#     # conf_df = pd.DataFrame(conf_matrix,
#     #    index=["Actual Non_tortuous", "Actual Tortuous"],
#     #    columns=["Predicted Non_tortuous", "Predicted Tortuous"]
#     # )
#     # conf_df.to_csv(f"./OCTA_tortuousity/model_eval/final_csvs/confusion_matrices/conf_matrix_{i}_thr_{best_threshold}.csv")

print("\n======== Done! =========")
