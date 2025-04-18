from ..model_training.model_config import  f1_score
import os
import glob
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
from .visualize_img import get_visualized_img
from .calculate_dice import calculate_dice_score
from .config import data_dirs, original_img_files, THRESHOLD_TIS
from .paper_replication_files.get_TI import ti
import pandas as pd
from tqdm import tqdm

weighted_dice_scores_final = []
dice_scores_final = []

ti_cache = []
for data_dir in data_dirs:
    tortuous_paths = glob.glob(os.path.join(data_dir, "tortuous", "*"))
    non_tortuous_paths = glob.glob(os.path.join(data_dir, "non_tortuous", "*"))
    all_image_paths = tortuous_paths + non_tortuous_paths
    for img_path in all_image_paths:
        ti_cache.append(ti(img_path))  # Calculate TI once

# THRESHOLD_TI = THRESHOLD_TIS[0]
# Process each dataset
max_dice = 0
max_weighted_dice = 0
max_thresh = 0
for THRESHOLD_TI in THRESHOLD_TIS:
    print("=" * 50)
    print("FOR TI THRESHOLD", THRESHOLD_TI)
    print("=" * 50)
    
    # Initialize the dataframe to store results
    metrics_df = pd.DataFrame(columns=[
        'Directory', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Dice Score', 'Weighted Dice Score'
    ])
    accuracies, precisions, recalls, f1_scores, dice_indexes = [], [], [], [], []

    weights_for_image = [] # is needed for the average dice score calculation (we store the number of pixels in the ground truth)


    confusion_matrices = []  # List to store confusion matrices for each dataset
    dice_scores = []  # Store dice scores
    dirs_for_conf = [] # directory for name of confusion matrix file
    weighted_dice_score = []
    weights_for_image = []
    # Process each dataset
    curr_ti_index = 0
    for i, data_dir in enumerate(data_dirs):
        # get the folder name for saving wrong images
        image_dir_name = os.path.normpath(data_dir)
        image_dir_name = image_dir_name.split("\\")[-2]
        

        # Gather image paths and labels
        tortuous_paths = glob.glob(os.path.join(data_dir, "tortuous", "*"))
        non_tortuous_paths = glob.glob(os.path.join(data_dir, "non_tortuous", "*"))
        all_image_paths = tortuous_paths + non_tortuous_paths
        all_labels = [1] * len(tortuous_paths) + [0] * len(non_tortuous_paths)

        # Initialize variables for metrics
        tp, tn, fp, fn = 0, 0, 0, 0
        pred_prob, pred_labels, predicted_tortuous_paths = [], [], []
        ti_indices = []
        pred_labels = []
        # Process images in batches
        for idx, image_path in tqdm(enumerate(all_image_paths), total=len(all_image_paths), desc="Processing and predicting images"):
            try:
                # img = load_and_pad_image(image_path, 224, True)
                # print(image_path)
                # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                # ti_of_img = ti(image_path)
                ti_of_img = ti_cache[curr_ti_index]
                ti_indices.append(ti_of_img)
                
                # Process the prediction
                pred_label = 1 if ti_of_img > THRESHOLD_TI else 0
                pred_labels.append(pred_label)

                if pred_label == 1:
                    # print(batch_image_path[j])
                    predicted_tortuous_paths.append(all_image_paths[idx])
                
                true_label = all_labels[idx]
                # Update confusion matrix counters
                if pred_label == 1 and true_label == 1:
                    tp += 1
                elif pred_label == 0 and true_label == 0:
                    tn += 1
                elif pred_label == 1 and true_label == 0:
                    #is non tortuous but predicted tortuous
                    fp += 1
                elif pred_label == 0 and true_label == 1:
                    #is tortuous but predicted non tortuous
                    fn += 1

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue
            curr_ti_index += 1

        # Compute metrics
        accuracy = accuracy_score(all_labels, pred_labels)
        precision = precision_score(all_labels, pred_labels)
        recall = recall_score(all_labels, pred_labels)
        f1 = f1_score(all_labels, pred_labels)

        report = classification_report(all_labels, pred_labels, target_names=['Non-Tortuous', 'Tortuous'])
        
        print("=" * 50)
        print(f"For directory: {data_dir}")
        print("=" * 50)
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
            ground, weight = get_visualized_img(original_img_files[i], tortuous_paths, is_prediction=False, save_image=False)
        if predicted_tortuous_paths:
            predicted, _ = get_visualized_img(original_img_files[i], predicted_tortuous_paths, is_prediction=True, save_image=False)

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


    # Compute and save weighted dice score
    weights_sum = sum(weights_for_image)
    if weights_sum == 0:
        weighted_dice_score = 0
    else:
        weighted_dice_score = np.sum(np.array(weights_for_image) * np.array(dice_scores)) / sum(weights_for_image) 

    if max_weighted_dice < weighted_dice_score:
        max_weighted_dice = weighted_dice_score
        max_thresh = THRESHOLD_TI
    metrics_df['Weighted Dice Score'] = weighted_dice_score
    final_dice = sum(dice_scores) / len(dice_scores)
    print(f"Dice Score Final: {final_dice}")
    dice_scores_final.append(final_dice)
    print(f"Weighted Final Dice Score: {weighted_dice_score}")
    weighted_dice_scores_final.append(weighted_dice_score)

    print("=" * 50)

    # Save metrics to CSV
    metrics_df.to_csv('./OCTA_tortuousity/model_eval/final_csvs/final_paper_results.csv', index=False)

    # Save confusion matrices to CSV (optional)
    for idx, conf_matrix in enumerate(confusion_matrices):
        print(dirs_for_conf[idx])
        # assumes that the path is
        # model_eval/images\{file_name}\result
        # conf_matrix.to_csv(f"./OCTA_tortuousity/model_eval/final_csvs/confusion_matrices/confusion_matrix_{idx}.csv", index=True)
        
    print("=" * 50)

print(f"Max weighted dice is : {max_weighted_dice} for Threshold TI of {max_thresh}")
print("final metrics are")
print(THRESHOLD_TIS)
print(weighted_dice_scores_final)
print(dice_scores_final)