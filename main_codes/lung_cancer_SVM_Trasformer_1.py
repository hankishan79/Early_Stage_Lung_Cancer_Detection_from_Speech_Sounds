import numpy as np
import pandas as pd
import torch
from featurewiz import FeatureWiz
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch import nn, optim
import matplotlib.pyplot as plt

def myround(x, base, div, subt):
    return (base * round(x / base) / div) - subt

data = pd.read_csv(r'..\dataset\feature_dataset.csv')
data.head()

y = data['CLASSES']
X = data.drop(columns=['CLASSES', 'Segment', 'Stage', 'PAT_ID'], axis=1)

X['KURTOSIS'] = pd.to_numeric(X['KURTOSIS'], errors='coerce')
X['KURTOSIS_f'] = pd.to_numeric(X['KURTOSIS_f'], errors='coerce')
X['SKEW'] = pd.to_numeric(X['SKEW'], errors='coerce')
X['SKEW_f'] = pd.to_numeric(X['SKEW_f'], errors='coerce')

# Feature transformation and selection (using FeatureWiz)
features = FeatureWiz(corr_limit=0.95, feature_engg='', category_encoders='', dask_xgboost_flag=False, nrows=None, verbose=2)
X = features.fit_transform(X, y)
print(X.dtypes)

# Drop rows with NaN values (optional, depending on your data and goals)
X = X.dropna()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVC model for classification
classifier_svm = SVC(kernel='rbf', C=10, gamma='scale')

# Fit the SVM model on the training data
classifier_svm.fit(X_train, y_train)

# Predict on the testing data using the SVM model
y_pred_svm = classifier_svm.decision_function(X_test)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

# Create the DiffusionTransformerClassification model
class DiffusionTransformerClassification(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DiffusionTransformerClassification, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=1)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x[-1])
        return torch.sigmoid(x)

# Create and train the DiffusionTransformerClassification model
model_transformer = DiffusionTransformerClassification(input_size=X.shape[1], hidden_size=32, output_size=1)  # Reduced hidden_size
criterion = nn.BCELoss()
optimizer_transformer = optim.Adam(model_transformer.parameters(), lr=0.01)
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_transformer.to(device)

# Convert pandas DataFrame to PyTorch tensor
X_train_tensor = torch.from_numpy(X_train.values).to(device).float()
y_train_tensor = torch.from_numpy(y_train.values).to(device).float().view(-1)  # Remove the extra dimension

# Train the model
batch_size = 16
for epoch in range(10):
    for i in range(0, len(X_train_tensor), batch_size):
        optimizer_transformer.zero_grad()
        output_transformer = model_transformer(X_train_tensor[i:i+batch_size].unsqueeze(0))  # Unsqueezed for batch dimension
        loss_transformer = criterion(output_transformer.view(-1), y_train_tensor[i:i+batch_size])
        loss_transformer.backward()
        optimizer_transformer.step()
    torch.cuda.empty_cache()  # Release GPU memory

# Move test data to the same device as the model
X_test_tensor = torch.from_numpy(X_test.values).to(device).float()
y_test_tensor = torch.from_numpy(y_test.values).to(device).float().view(-1, 1)

# After training, you can use the model for predictions
model_transformer.eval()
with torch.no_grad():
    y_probs_transformer = model_transformer(X_test_tensor.unsqueeze(0)).cpu().numpy()  # Unsqueezed for batch dimension

# Calculate ROC AUC for the Transformer model
fpr_transformer, tpr_transformer, _ = roc_curve(y_test, y_probs_transformer)
roc_auc_transformer = auc(fpr_transformer, tpr_transformer)

# Plot ROC Curves for all three models
plt.figure(figsize=(10, 6))
plt.plot(fpr_svm, tpr_svm, color='darkorange', lw=2, label=f'SVM (AUC = {roc_auc_svm:.2f})')
plt.plot(fpr_transformer, tpr_transformer, color='blue', lw=2, label=f'Transformer (AUC = {roc_auc_transformer:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
