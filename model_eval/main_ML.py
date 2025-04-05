import os
import glob
import joblib  # For loading the .pkl models
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
from .config import data_dirs, original_img_files, PROB_THRESHOLD
from .visualize_img import get_visualized_img
from .calculate_dice import calculate_dice_score
from ..model_training.ML_test.process_images import process_images
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from collections import defaultdict

# Initialize the dataframe to store results
metrics_df = pd.DataFrame(columns=[
    'Directory', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Dice Score', 'Weighted Dice Score'
])

confusion_matrices = []  # List to store confusion matrices for each dataset
dirs_for_conf = []  # Directories for the confusion matrix files
# Load models from .pkl files
model_files = glob.glob('./OCTA_tortuousity/models/*.pkl')  # Adjust the path based on where your models are stored
models = {}
dice_scores = defaultdict(list)  # Format: {model: {data_dir: dice}}
data_weights = defaultdict(int)  # Format: {data_dir: weight}
print(f"Total {len(model_files)} models loaded")

for model_file in model_files:
    model_name = os.path.basename(model_file).split(".")[0]
    try:
        models[model_name] = joblib.load(model_file)
        print(f"Loaded model: {model_name}")
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")

# Process each dataset
# print(data_dirs)
for i, data_dir in enumerate(data_dirs):
    print(f"For data dir {data_dir}")
    
    # Gather image paths and labels
    tortuous_paths = glob.glob(os.path.join(data_dir, "tortuous", "*"))
    non_tortuous_paths = glob.glob(os.path.join(data_dir, "non_tortuous", "*"))
    all_image_paths = tortuous_paths + non_tortuous_paths
    all_labels = [1] * len(tortuous_paths) + [0] * len(non_tortuous_paths)
    
    # Process images using the external function `process_images` to extract features
    features_df = process_images(data_dir)
    print(features_df.head())
    # print(features_df.head())
    # Loop through each model and compute predictions
    data_weights_processed = False
    for model_name, model in models.items():
        print(f"Processing with model: {model_name}")

        # Initialize variables for metrics
        predicted_tortuous_paths = []
        tp, tn, fp, fn = 0, 0, 0, 0    
        
        # Initialize lists for batching
        batch_images = features_df.values  # Assuming the features are in the DataFrame
        batch_labels = all_labels

        # Make predictions for the batch
        # preds = model.predict(batch_images)
        preds = []
        if hasattr(model, 'predict_proba'):
            preds = model.predict_proba(batch_images)[:, 1]  # Get probabilities for the positive class
        else:
            preds = model.decision_function(batch_images)

        # Initialize lists for metrics
        model_pred_labels = []
        model_pred_prob = []

        # Process each prediction
        for j, (pred, true_label) in enumerate(zip(preds, batch_labels)):
            model_pred_prob.append(pred)
            pred_label = 1 if pred > PROB_THRESHOLD else 0
            model_pred_labels.append(pred_label)

            if pred_label == 1:
                predicted_tortuous_paths.append(all_image_paths[j])

            # Update confusion matrix counters
            if pred_label == 1 and true_label == 1:
                tp += 1
            elif pred_label == 0 and true_label == 0:
                tn += 1
            elif pred_label == 1 and true_label == 0:
                # Save wrongly predicted tortuous images
                filename = all_image_paths[j].split(".")[0].split("\\")[-1]
                # if not os.path.exists(f"model_eval/wronglyTortuous/{model_name}"):
                #     os.makedirs(f"model_eval/wronglyTortuous/{model_name}")
                # cv2.imwrite(f"model_eval/wronglyTortuous/{model_name}/{filename}.png", features_df.iloc[j].values.reshape(224, 224) * 255)  # Adjust shape if needed
                fp += 1
            elif pred_label == 0 and true_label == 1:
                # Save wrongly predicted non-tortuous images
                filename = all_image_paths[j].split(".")[0].split("\\")[-1]
                # if not os.path.exists(f"model_eval/wronglyNonTortuous/{model_name}"):
                #     os.makedirs(f"model_eval/wronglyNonTortuous/{model_name}")
                # cv2.imwrite(f"model_eval/wronglyNonTortuous/{model_name}/{filename}.png", features_df.iloc[j].values.reshape(224, 224) * 255)  # Adjust shape if needed
                fn += 1

        # Compute metrics
        accuracy = accuracy_score(all_labels, model_pred_labels)
        precision = precision_score(all_labels, model_pred_labels)
        recall = recall_score(all_labels, model_pred_labels)
        f1 = f1_score(all_labels, model_pred_labels)

        # Save results in the dataframe for this model
        metrics_df = pd.concat([metrics_df, pd.DataFrame([{
            'Directory': data_dir,
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Dice Score': None,  # Dice Score to be calculated later
            'Weighted Dice Score': None  # Weighted Dice Score to be calculated later
        }])], ignore_index=True)

        # Print results
        print("=" * 50)
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Print confusion matrix
        conf_matrix = confusion_matrix(all_labels, model_pred_labels)
        conf_matrix = pd.DataFrame(conf_matrix, index=["Actual Non_tortuous", "Actual Tortuous"], columns=["Predicted Non_tortuous", "Predicted Tortuous"])
        print("Confusion Matrix:")
        print(conf_matrix)

        dirs_for_conf.append(data_dir)
        confusion_matrices.append({
            'model': model_name,
            'data_dir': data_dir,
            'matrix': conf_matrix
        })


        # Visualize results (using the same visualizing function as before)
        ground = []
        weight = 0
        predicted = []

        if tortuous_paths:
            ground, weight = get_visualized_img(original_img_files[i], tortuous_paths, is_prediction=False, save_image = False)
            if not data_weights_processed:
                data_weights[data_dir] += weight
                data_weights_processed = True # need to get the weights only once per dir so, no need to repeat it after one model
        if predicted_tortuous_paths:
            predicted, _ = get_visualized_img(original_img_files[i], predicted_tortuous_paths, is_prediction=True, save_image = False)
        # print("ground is ", predicted)
        dice_index = calculate_dice_score(ground, predicted)
        # dice_scores.append(dice_index)
        dice_scores[model_name].append(dice_index)


        # Update the dataframe with dice score
        metrics_df.at[len(metrics_df)-1, 'Dice Score'] = dice_index

        print(f"Dice Score is: {dice_index}")
        print("=" * 50)

# Final statistics for all models
print("Final Prediction Stats:")
if len(metrics_df) > 0:
    print(f"Accuracy: {metrics_df['Accuracy'].mean():.4f}")
    print(f"Precision: {metrics_df['Precision'].mean():.4f}")
    print(f"Recall: {metrics_df['Recall'].mean():.4f}")
    print(f"F1 Score: {metrics_df['F1 Score'].mean():.4f}")


# Save metrics to CSV for each model
# **********Note that here we assume an ordered dictionary for correct calculations***************
weights = list(data_weights.values())
sum_weights = np.sum(weights)
for model_name in models:
    model_dice = dice_scores[model_name]
    weighted_dice_score = np.sum(np.array(model_dice) * np.array(weights)) / sum_weights
    metrics_df['Weighted Dice Score'] = weighted_dice_score

    model_metrics_df = metrics_df[metrics_df['Model'] == model_name]
    print(f"Weighted Final Dice Score for {model_name} is : {weighted_dice_score}")
    model_metrics_df.to_csv(f'./OCTA_tortuousity/model_eval/final_csvs/{model_name}_results.csv', index=False)

# Save confusion matrices to CSV for each model (optional)
# print(len(confusion_matrices))
# for idx, conf_matrix in enumerate(confusion_matrices):
#     print(dirs_for_conf[idx])
#     conf_matrix.to_csv(f"./OCTA_tortuousity/model_eval/final_csvs/confusion_matrix_{list(models.keys())[idx]}.csv", index=True)

print("=" * 50)
