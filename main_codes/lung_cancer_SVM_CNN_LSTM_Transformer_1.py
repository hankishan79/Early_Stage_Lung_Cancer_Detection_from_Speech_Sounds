import numpy as np
import pandas as pd
import torch
from featurewiz import FeatureWiz
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, LSTM
from keras.utils import to_categorical
from torch import nn, optim


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

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert target labels to one-hot encoding
y_encoded = to_categorical(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Create an SVC model for classification
svm_classifier = SVC(kernel='rbf', C=10, gamma='scale')

# Fit the SVM model on the training data
svm_classifier.fit(X_train, y_train.argmax(axis=1))

# Predict on the testing data
y_pred_svm = svm_classifier.predict(X_test)

# Calculate and print accuracy, classification report, precision, recall, F1 score, and ROC AUC for SVM
accuracy_svm = accuracy_score(y_test.argmax(axis=1), y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

print("SVM Classification Report:\n", classification_report(y_test.argmax(axis=1), y_pred_svm))
print("SVM Precision:", precision_score(y_test.argmax(axis=1), y_pred_svm, average='weighted'))
print("SVM Recall:", recall_score(y_test.argmax(axis=1), y_pred_svm, average='weighted'))
print("SVM F1 Score:", f1_score(y_test.argmax(axis=1), y_pred_svm, average='weighted'))

# Confusion Matrix for SVM
conf_matrix_svm = confusion_matrix(y_test.argmax(axis=1), y_pred_svm)
print("SVM Confusion Matrix:\n", conf_matrix_svm)

# Calculate ROC AUC for SVM
y_probs_svm = svm_classifier.decision_function(X_test)
fpr_svm, tpr_svm, _ = roc_curve(y_test.argmax(axis=1), y_probs_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)
print("SVM ROC AUC:", roc_auc_svm)

# Create a simple CNN model
cnn_model = Sequential()
cnn_model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dense(len(np.unique(y)), activation='softmax'))  # Output layer

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Reshape X_train and X_test to add a channel dimension
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Fit the CNN model on the training data
cnn_model.fit(X_train_reshaped, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

# Predict on the testing data
y_probs_cnn = cnn_model.predict(X_test_reshaped)[:, 1]
y_pred_cnn = np.argmax(cnn_model.predict(X_test_reshaped), axis=1)

# Calculate and print accuracy, classification report, precision, recall, F1 score, and ROC AUC for CNN
accuracy_cnn = accuracy_score(y_test.argmax(axis=1), y_pred_cnn)
print("CNN Accuracy:", accuracy_cnn)

print("CNN Classification Report:\n", classification_report(y_test.argmax(axis=1), y_pred_cnn))
print("CNN Precision:", precision_score(y_test.argmax(axis=1), y_pred_cnn, average='weighted'))
print("CNN Recall:", recall_score(y_test.argmax(axis=1), y_pred_cnn, average='weighted'))
print("CNN F1 Score:", f1_score(y_test.argmax(axis=1), y_pred_cnn, average='weighted'))

# Confusion Matrix for CNN
conf_matrix_cnn = confusion_matrix(y_test.argmax(axis=1), y_pred_cnn)
print("CNN Confusion Matrix:\n", conf_matrix_cnn)

# Calculate ROC AUC for CNN
fpr_cnn, tpr_cnn, _ = roc_curve(y_test.argmax(axis=1), y_probs_cnn)
roc_auc_cnn = auc(fpr_cnn, tpr_cnn)
print("CNN ROC AUC:", roc_auc_cnn)

# Create a simple LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(32, activation='relu', input_shape=(X_train.shape[1], 1)))
lstm_model.add(Dense(len(np.unique(y)), activation='softmax'))  # Output layer

lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the LSTM model on the training data
lstm_model.fit(X_train_reshaped, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

# Predict on the testing data
y_probs_lstm = lstm_model.predict(X_test_reshaped)[:, 1]
y_pred_lstm = np.argmax(lstm_model.predict(X_test_reshaped), axis=1)

# Calculate and print accuracy, classification report, precision, recall, F1 score, and ROC AUC for LSTM
accuracy_lstm = accuracy_score(y_test.argmax(axis=1), y_pred_lstm)
print("LSTM Accuracy:", accuracy_lstm)

print("LSTM Classification Report:\n", classification_report(y_test.argmax(axis=1), y_pred_lstm))
print("LSTM Precision:", precision_score(y_test.argmax(axis=1), y_pred_lstm, average='weighted'))
print("LSTM Recall:", recall_score(y_test.argmax(axis=1), y_pred_lstm, average='weighted'))
print("LSTM F1 Score:", f1_score(y_test.argmax(axis=1), y_pred_lstm, average='weighted'))
# Confusion Matrix for CNN
conf_matrix_lstm = confusion_matrix(y_test.argmax(axis=1), y_pred_lstm)
print("LSTM Confusion Matrix:\n", conf_matrix_lstm)

# Calculate ROC AUC for CNN
fpr_lstm, tpr_lstm, _ = roc_curve(y_test.argmax(axis=1), y_probs_lstm)
roc_auc_lstm = auc(fpr_lstm, tpr_lstm)
print("LSTM ROC AUC:", roc_auc_lstm)

# Create the DiffusionTransformerClassification model
class DiffusionTransformerClassification(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DiffusionTransformerClassification, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=1)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.fc = nn.Linear(input_size, 1)  # Updated output size to 1 for binary classification

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x[-1])
        return x  # Removed sigmoid here, it will be applied later during the loss calculation


# Create and train the DiffusionTransformerClassification model
model_transformer = DiffusionTransformerClassification(input_size=X.shape[1], hidden_size=32, output_size=1)  # Reduced hidden_size
criterion = nn.BCELoss()
optimizer_transformer = optim.Adam(model_transformer.parameters(), lr=0.01)
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_transformer.to(device)
# Convert pandas DataFrame to PyTorch tensor
X_train_tensor = torch.from_numpy(X_train).to(device).float()
y_train_tensor = torch.from_numpy(y_train).to(device).float().view(-1)  # Remove the extra dimension
# Train the model
batch_size = 16
# Inside the training loop
# Inside the training loop
for i in range(0, len(X_train_tensor), batch_size):
    optimizer_transformer.zero_grad()

    output_transformer = model_transformer(X_train_tensor[i:i + batch_size].unsqueeze(0))  # Unsqueezed for batch dimension

    # Apply sigmoid to ensure outputs are between 0 and 1
    output_transformer = torch.sigmoid(output_transformer)

    # Correctly sized target tensor
    target = y_train_tensor[i:i + batch_size].unsqueeze(1).float()  # Unsqueezed for target dimension and converted to float

    # Resize the output tensor to match the target size
    output_transformer = output_transformer[:, :target.size(1)]

    # Loss calculation
    loss_transformer = criterion(output_transformer, target)
    loss_transformer.backward()
    optimizer_transformer.step()


# Move test data to the same device as the model
X_test_tensor = torch.from_numpy(X_test).to(device).float()
y_test_tensor = torch.from_numpy(y_test).to(device).float().view(-1, 1)


# After training, you can use the model for predictions
model_transformer.eval()
with torch.no_grad():
    y_probs_transformer = model_transformer(X_test_tensor.unsqueeze(0)).cpu().numpy()  # Unsqueezed for batch dimension

# Calculate ROC AUC for the Transformer model
fpr_transformer, tpr_transformer, _ = roc_curve(y_test, y_probs_transformer)
roc_auc_transformer = auc(fpr_transformer, tpr_transformer)
print("Transformer ROC AUC:", roc_auc_transformer)

# Calculate and print accuracy, classification report, precision, recall, F1 score, and ROC AUC for Transformer
y_pred_transformer = np.round(y_probs_transformer)
accuracy_transformer = accuracy_score(y_test, y_pred_transformer)
precision_transformer = precision_score(y_test, y_pred_transformer)
recall_transformer = recall_score(y_test, y_pred_transformer)
f1_transformer = f1_score(y_test, y_pred_transformer)
print("Transformer Accuracy:", accuracy_transformer)
print("Transformer Precision:", precision_transformer)
print("Transformer Recall:", recall_transformer)
print("Transformer F1 Score:", f1_transformer)

# Plot combined ROC Curve
plt.figure(figsize=(12, 8))
plt.plot(fpr_svm, tpr_svm, color='blue', lw=2, label=f'SVM ROC curve (AUC = {roc_auc_svm:.2f})')
plt.plot(fpr_cnn, tpr_cnn, color='red', lw=2, label=f'CNN ROC curve (AUC = {roc_auc_cnn:.2f})')
plt.plot(fpr_lstm, tpr_lstm, color='green', lw=2, label=f'LSTM ROC curve (AUC = {roc_auc_lstm:.2f})')
plt.plot(fpr_transformer, tpr_transformer, color='purple', lw=2, label=f'Transformer ROC curve (AUC = {roc_auc_transformer:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - SVM vs CNN vs LSTM vs Transformer')
plt.legend(loc="lower right")
plt.show()

# Perform 10-fold cross-validation for SVM
kfold_svm = KFold(n_splits=10, shuffle=True, random_state=42)
scores_svm = cross_val_score(svm_classifier, X_scaled, y, cv=kfold_svm, scoring='accuracy')

# Print the accuracy for each fold for SVM
for fold, score in enumerate(scores_svm):
    print(f"SVM Fold {fold+1}: {score}")

# Print the average accuracy and standard deviation for SVM
print(f"SVM Average Accuracy: {np.mean(scores_svm)}")
print(f"SVM Standard deviation: {np.std(scores_svm)}")

# Perform 10-fold cross-validation for CNN
kfold_cnn = KFold(n_splits=10, shuffle=True, random_state=42)
scores_cnn = cross_val_score(cnn_model, X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1), y_encoded, cv=kfold_cnn)

# Print the accuracy for each fold for CNN
for fold, score in enumerate(scores_cnn):
    print(f"CNN Fold {fold+1}: {score}")

# Print the average accuracy and standard deviation for CNN
print(f"CNN Average Accuracy: {np.mean(scores_cnn)}")
print(f"CNN Standard deviation: {np.std(scores_cnn)}")

# Perform 10-fold cross-validation for LSTM
kfold_lstm = KFold(n_splits=10,shuffle=True, random_state=42)
scores_lstm = cross_val_score(lstm_model, X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1), y_encoded, cv=kfold_lstm)

# Print the accuracy for each fold for CNN
for fold, score in enumerate(scores_lstm):
    print(f"LSTM Fold {fold+1}: {score}")

# Perform k-fold cross-validation for Transformer
kfold_transformer = KFold(n_splits=10, shuffle=True, random_state=42)
scores_transformer = cross_val_score(model_transformer, X_scaled, y, cv=kfold_transformer, scoring='accuracy')

# Print the accuracy for each fold for Transformer
for fold, score in enumerate(scores_transformer):
    print(f"Transformer Fold {fold+1}: {score}")

# Print the average accuracy and standard deviation for Transformer
print(f"Transformer Average Accuracy: {np.mean(scores_transformer)}")
print(f"Transformer Standard deviation: {np.std(scores_transformer)}")
