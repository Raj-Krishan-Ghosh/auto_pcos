import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,roc_curve
import matplotlib.pyplot as plt

# Load the ground truth and validation predicted Excel sheets
class_label_df = pd.read_excel(r'Dataset/PCOSGen-train/class_label.xlsx')
val_pred_df = pd.read_excel(r'train_val/val_predicted/val_predicted.xlsx')

# only including prediction images in ground truth df
class_label_df = class_label_df[class_label_df['imagePath'].isin(val_pred_df["Image Path (in ascending order)"].to_list())]

true_labels = class_label_df["Healthy"].to_numpy()
predicted_labels = val_pred_df["Predicted class label"].to_numpy()

# Merge the two dataframes on image_path
#merged_df = pd.merge(class_label_df, val_pred_df, on='image_path')

# Extract true labels and predicted labels
#true_labels = merged_df['class_label']
#predicted_labels = merged_df['Predicted_class_label']

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
auc_roc = roc_auc_score(true_labels, predicted_labels)

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC-ROC Score: {auc_roc}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)
plt.plot(fpr, tpr, label=f'AUC-ROC = {auc_roc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()