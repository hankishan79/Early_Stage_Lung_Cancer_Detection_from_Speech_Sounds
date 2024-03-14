import numpy as np
import pandas as pd
import torch
from featurewiz import FeatureWiz
from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, precision_score, recall_score, \
    f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import pandas as pd
from featurewiz import FeatureWiz
import matplotlib.pyplot as plt
from torch_geometric.utils import add_self_loops
from torchvision import transforms
from PIL import Image  # Add import for PIL Image

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
X_scaled=X
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an SVC model for classification
classifier_svm = SVC(kernel='rbf', C=.1, gamma='scale')

# Fit the SVM model on the training data
classifier_svm.fit(X_train, y_train)
# Predict on the testing data
y_pred_svm = classifier_svm.predict(X_test)
print("svm ok")
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
model_transformer = DiffusionTransformerClassification(input_size=X.shape[1], hidden_size=16, output_size=1)  # Reduced hidden_size
criterion = nn.BCELoss()
optimizer_transformer = optim.AdamW(model_transformer.parameters(), lr=0.01)
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_transformer.to(device)

# Convert pandas DataFrame to PyTorch tensor
X_train_tensor = torch.from_numpy(X_train).to(device).float()
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
X_test_tensor = torch.from_numpy(X_test).to(device).float()
y_test_tensor = torch.from_numpy(y_test.values).to(device).float().view(-1, 1)

# After training, you can use the model for predictions
model_transformer.eval()
with torch.no_grad():
    y_probs_transformer = model_transformer(X_test_tensor.unsqueeze(0)).cpu().numpy()  # Unsqueezed for batch dimension
print("transformer ok")
#*************************CNN********************************************************
# Create a simple CNN model
cnn_model = Sequential()
cnn_model.add(Conv1D(8, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())
cnn_model.add(Dense(16, activation='relu'))
cnn_model.add(Dense(len(np.unique(y)), activation='softmax'))  # Output layer
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Reshape X_train and X_test to add a channel dimension
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
y_train_encoded = to_categorical(y_train)
# Fit the CNN model on the training data
cnn_model.fit(X_train_reshaped, y_train_encoded, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
# Predict on the testing data
y_probs_cnn = cnn_model.predict(X_test_reshaped)[:, 1]
y_pred_cnn = np.argmax(cnn_model.predict(X_test_reshaped), axis=1)

#************************LSTM**********************************************************************
# Create a simple LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(32, activation='relu', input_shape=(X_train.shape[1], 1)))
lstm_model.add(Dense(len(np.unique(y)), activation='softmax'))  # Output layer
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Fit the LSTM model on the training data
lstm_model.fit(X_train_reshaped, y_train_encoded, epochs=30, batch_size=32, validation_split=0.2, verbose=1)
# Predict on the testing data
y_probs_lstm = lstm_model.predict(X_test_reshaped)[:, 1]
y_pred_lstm = np.argmax(lstm_model.predict(X_test_reshaped), axis=1)

#********************LSTM-CNN HYBRID ********************************************************
# Create a simple LSTM-CNN hybrid model
hybrid_model = Sequential()
# CNN layer
hybrid_model.add(Conv1D(8, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
hybrid_model.add(MaxPooling1D(pool_size=2))
# LSTM layer
hybrid_model.add(LSTM(16, activation='relu'))
# Dense layer (output layer)
hybrid_model.add(Dense(len(np.unique(y)), activation='softmax'))
# Compile the model
hybrid_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
# Display the model summary
hybrid_model.summary()
# Fit the hybrid model on the training data
hybrid_model.fit(X_train_reshaped, y_train_encoded, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
# Predict on the testing data
y_probs_hybrid = hybrid_model.predict(X_test_reshaped)[:, 1]
y_pred_hybrid = np.argmax(hybrid_model.predict(X_test_reshaped), axis=1)

#**********************GCN ***********************************************
class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)
# Create and train the GCN model
model_gcn = GCN(input_size=X.shape[1], hidden_size=64, output_size=1)
criterion_gcn = nn.BCELoss()
optimizer_gcn = optim.AdamW(model_gcn.parameters(), lr=0.01)
# Move model to GPU if available
model_gcn.to(device)
# Assuming X_scaled.shape[0] is the number of nodes in your graph
num_nodes = X_train.shape[0]
# Create edge_index using torch_geometric.utils.add_self_loops
edge_index = add_self_loops(torch.zeros(2, num_nodes, dtype=torch.long), num_nodes=num_nodes)[0]
# Convert pandas DataFrame to PyTorch tensor
X_train_gcn = torch.from_numpy(X_train).to(device).float()
#edge_index = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.long).to(device)
# Train the GCN model
for epoch in range(150):
    optimizer_gcn.zero_grad()
    output_gcn = model_gcn(X_train_gcn, edge_index)
    loss_gcn = criterion_gcn(output_gcn.view(-1), y_train_tensor)
    loss_gcn.backward()
    optimizer_gcn.step()
torch.cuda.empty_cache()  # Release GPU memory
# Move test data to the same device as the model
X_test_gcn = torch.from_numpy(X_test).to(device).float()
# After training, you can use the GCN model for predictions
model_gcn.eval()
with torch.no_grad():
    y_probs_gcn = model_gcn(X_test_gcn, edge_index).cpu().numpy()
#***********************GAT-CL*****************************************
import torch
import torchvision.transforms.functional as F
import numpy as np
from scipy.ndimage import convolve1d

class CustomJitter:
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, img):
        img = np.array(img)
        jitter = np.random.normal(loc=0, scale=self.sigma, size=img.shape)
        jittered_img = img + jitter
        return Image.fromarray(jittered_img.astype('uint8'))

class CustomPermutation:
    def __call__(self, img):
        img = np.array(img)
        permutation = np.random.permutation(img.flatten()).reshape(img.shape)
        return Image.fromarray(permutation.astype('uint8'))

class CustomTimeWarping:
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, img):
        img = np.array(img)
        h, w = img.shape
        control_points = np.linspace(0, w - 1, num=5)  # Use 'w' instead of 'h' for control points

        # Generate 1D displacement for each row
        displacement = np.random.uniform(-self.alpha, self.alpha, size=(h, len(control_points)))

        # Apply time warping to each row separately
        for i in range(h):
            # Ensure the length of interpolated_displacement matches the width (w) of the image
            interpolated_displacement = np.interp(np.arange(w), control_points, displacement[i])
            img[i, :] = np.clip(img[i, :] + interpolated_displacement, 0, 255).astype('uint8')

        return Image.fromarray(img)

class CustomScaling:
    def __init__(self, scale_factor_range=(0.8, 1.2)):
        self.scale_factor_range = scale_factor_range

    def __call__(self, img):
        # Check if the image size is zero and return the original image
        if img.size[0] <= 0 or img.size[1] <= 0:
            return img

        scale_factor = np.random.uniform(*self.scale_factor_range)

        # Check if after resizing the image will have dimensions greater than 0
        new_size = [int(dim * scale_factor) for dim in img.size]
        if new_size[0] <= 0 or new_size[1] <= 0:
            return img

        img = F.resize(img, new_size)
        return img

class CustomInversion:
    def __call__(self, img):
        img = np.array(img)
        inverted_img = 255 - img
        return Image.fromarray(inverted_img.astype('uint8'))

class CustomLowPassFilter:
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, img):
        img = np.array(img)
        smoothed_img = convolve1d(img, weights=np.exp(-np.arange(-5, 6)**2 / (2 * self.sigma**2)))
        return Image.fromarray(smoothed_img.astype('uint8'))

class CustomPhaseShift:
    def __init__(self, shift_range=(0, 2*np.pi)):
        self.shift_range = shift_range

    def __call__(self, img):
        img = np.array(img)
        phase_shift = np.random.uniform(*self.shift_range)
        shifted_img = np.fft.ifft2(np.fft.fft2(img) * np.exp(1j * phase_shift)).real
        return Image.fromarray(shifted_img.astype('uint8'))

# Define data augmentation transformations
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    # Add more transformations as needed
    CustomJitter(),  # Custom transformation for jitter
    #CustomPermutation(),  # Custom transformation for permutation
    #CustomTimeWarping(),  # Custom transformation for time warping
    #CustomInversion(),  # Custom transformation for inversion
    #CustomLowPassFilter(),  # Custom transformation for low pass filter
    #CustomPhaseShift(),  # Custom transformation for phase shift
])
# Define the self-supervised contrastive learning model
class ContrastiveNet(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=128):
        super(ContrastiveNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.encoder(x)

# Instantiate the contrastive learning model
input_size = X_train.shape[1]
output_size = 128  # Adjust this based on your desired output size
contrastive_model = ContrastiveNet(input_size=input_size, output_size=output_size)
# GAT Layer
class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.5):  # Dropout ekleyin
        super(GATLayer, self).__init__()
        self.gat_conv = GATConv(in_channels, out_channels, heads=heads, concat=True, dropout=dropout)

    def forward(self, x, edge_index):
        return self.gat_conv(x, edge_index)
class TransformerLayer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super(TransformerLayer, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )

    def forward(self, x):
        return self.transformer(x)
class ContrastiveNetWithGAT(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=128, gat_hidden_size=64, gat_heads=4):
        super(ContrastiveNetWithGAT, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.gat_layer = GATLayer(output_size, gat_hidden_size, heads=gat_heads)

    def forward(self, x, edge_index):
        x = self.encoder(x)
        x = self.gat_layer(x, edge_index)
        return x
class ContrastiveNetWithGATAndTransformer(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=128, gat_hidden_size=64, gat_heads=4,
                 transformer_d_model=128, transformer_nhead=4, transformer_num_layers=2):
        super(ContrastiveNetWithGATAndTransformer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.gat_layer = GATLayer(output_size, gat_hidden_size, heads=gat_heads)
        self.transformer_layer = TransformerLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers
        )

    def forward(self, x, edge_index):
        x = self.encoder(x)
        x_gat = self.gat_layer(x, edge_index)
        x_transformer = self.transformer_layer(x_gat)
        return x_transformer
# Instantiate the contrastive learning model with GAT
gat_hidden_size = 64
gat_heads = 2
transformer_d_model = 128
transformer_nhead = 4
transformer_num_layers = 2

contrastive_model_with_gat_and_transformer = ContrastiveNetWithGATAndTransformer(
    input_size=input_size,
    output_size=output_size,
    gat_hidden_size=gat_hidden_size,
    gat_heads=gat_heads,
    transformer_d_model=transformer_d_model,
    transformer_nhead=transformer_nhead,
    transformer_num_layers=transformer_num_layers
)

# Define the contrastive loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.AdamW(contrastive_model_with_gat_and_transformer.parameters(), lr=0.001)
# Create a custom dataset for contrastive learning with data augmentation
class AugmentedContrastiveDataset(Dataset):
    def __init__(self, X, transform=None):
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        # Convert the tensor to a PIL image
        sample_pil = Image.fromarray(sample.numpy())
        if self.transform:
            sample_pil = self.transform(sample_pil)
        # Convert the PIL image to a tensor
        sample = transforms.ToTensor()(sample_pil)
        return sample
# Create datasets and dataloaders for contrastive learning with data augmentation
augmented_contrastive_dataset = AugmentedContrastiveDataset(X_train_tensor, transform=data_transform)
augmented_contrastive_loader = DataLoader(augmented_contrastive_dataset, batch_size=64, shuffle=True)

# Her `step_size` epokta bir öğrenme oranını `gamma` ile çarpacak bir scheduler oluşturun
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
# Train the contrastive learning model with GAT using augmented data
# Train the contrastive learning model with GAT using augmented data
num_epochs = 60

for epoch in range(num_epochs):
    for batch in augmented_contrastive_loader:
        inputs = batch
        optimizer.zero_grad()
        # Ensure the inputs are flattened to (batch_size, input_size)
        inputs = inputs.view(inputs.size(0), -1)
        outputs = contrastive_model_with_gat_and_transformer(inputs, edge_index)
        # Apply sigmoid activation
        outputs = torch.sigmoid(outputs)
        # Create target tensor with all zeros
        target = torch.zeros(outputs.shape[0], dtype=torch.float32)
        # Ensure the target tensor has the same size as the input tensor
        target = target.unsqueeze(1).expand_as(outputs)
        loss = criterion(outputs.squeeze(), target)  # Contrastive loss
        loss.backward()
        optimizer.step()
    # Scheduler'ı güncelleyin (her epoch sonunda)
    scheduler.step(epoch)  # Pass the current epoch to the scheduler

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Contrastive Loss: {loss.item():.4f}')

# Use the trained encoder with GAT for downstream classification
class DownstreamClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=2):
        super(DownstreamClassifier, self).__init__()
        self.encoder = contrastive_model_with_gat_and_transformer.encoder  # Use the trained encoder with GAT from contrastive learning
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

# Instantiate and train the downstream classifier
output_size_classifier = 2  # Adjust this based on your desired output size for classification
classifier_model = DownstreamClassifier(input_size=output_size, output_size=output_size_classifier)
criterion_classifier = nn.CrossEntropyLoss()
optimizer_classifier = optim.AdamW(classifier_model.parameters(), lr=0.001)

# Train the downstream classifier
num_epochs_classifier = 3500
for epoch in range(num_epochs_classifier):
    optimizer_classifier.zero_grad()
    # Forward pass
    outputs_classifier = classifier_model(X_train_tensor)
    # Assuming you have a binary classification problem
    # If you have more classes, adjust accordingly
    loss_classifier = criterion_classifier(outputs_classifier, y_train_tensor.long())
    loss_classifier.backward()
    optimizer_classifier.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs_classifier}], Classifier Loss: {loss_classifier.item():.4f}')

# Fine-tuning the downstream classifier
# Set a smaller learning rate for fine-tuning
fine_tune_lr = 0.0001
# Define a new optimizer for fine-tuning
optimizer_fine_tune = optim.AdamW(classifier_model.parameters(), lr=fine_tune_lr)
# Her `step_size` epokta bir öğrenme oranını `gamma` ile çarpacak bir scheduler oluşturun
scheduler_fine = StepLR(optimizer, step_size=10, gamma=0.5)

# Train the downstream classifier for additional epochs (fine-tuning)
num_epochs_fine_tune = 100
for epoch in range(num_epochs_fine_tune):
    optimizer_fine_tune.zero_grad()
    # Forward pass
    outputs_classifier = classifier_model(X_train_tensor)
    # Assuming you have a binary classification problem
    # If you have more classes, adjust accordingly
    loss_classifier = criterion_classifier(outputs_classifier, y_train_tensor.long())
    loss_classifier.backward()
    optimizer_fine_tune.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs_fine_tune}], Fine-tuned Classifier Loss: {loss_classifier.item():.4f}')

# Evaluate the fine-tuned classifier on the test set
with torch.no_grad():
    classifier_model.eval()
    outputs_test = classifier_model(X_test_tensor)
    outputs_test_roc = classifier_model(X_test_tensor)[:,1]
    _, predicted_labels = torch.max(outputs_test, 1)
    # Convert tensor to numpy array for sklearn metrics
    y_test_np = y_test_tensor.numpy()
    predicted_labels_np = predicted_labels.numpy()
    # Calculate metrics
    correct_predictions = (predicted_labels == y_test_tensor).sum().item()
    accuracy = correct_predictions / len(y_test_tensor)
    accuracy_cl = accuracy_score(y_test_tensor, predicted_labels)
    f1 = f1_score(y_test_np, predicted_labels_np)
    precision = precision_score(y_test_np, predicted_labels_np)
    recall = recall_score(y_test_np, predicted_labels_np)
    # For binary classification, use roc_auc_score
    auc2 = roc_auc_score(y_test_np, predicted_labels_np)

    encoded_train_data = contrastive_model_with_gat_and_transformer.encoder(X_train_tensor).detach().numpy()

#**********************************************************************
# Calculate and print accuracy, classification report, precision, recall, F1 score, and ROC AUC for SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))
print("SVM Precision:", precision_score(y_test, y_pred_svm, average='weighted'))
print("SVM Recall:", recall_score(y_test, y_pred_svm, average='weighted'))
print("SVM F1 Score:", f1_score(y_test, y_pred_svm, average='weighted'))
# Confusion Matrix for SVM
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
print("SVM Confusion Matrix:\n", conf_matrix_svm)
# Calculate ROC AUC for SVM
y_probs_svm = classifier_svm.decision_function(X_test)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_probs_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)
print("SVM ROC AUC:", roc_auc_svm)

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

# Calculate and print accuracy, classification report, precision, recall, F1 score, and ROC AUC for CNN
accuracy_cnn = accuracy_score(y_test, y_pred_cnn)
print("CNN Accuracy:", accuracy_cnn)
print("CNN Classification Report:\n", classification_report(y_test, y_pred_cnn))
print("CNN Precision:", precision_score(y_test, y_pred_cnn, average='weighted'))
print("CNN Recall:", recall_score(y_test, y_pred_cnn, average='weighted'))
print("CNN F1 Score:", f1_score(y_test, y_pred_cnn, average='weighted'))
# Confusion Matrix for CNN
conf_matrix_cnn = confusion_matrix(y_test, y_pred_cnn)
print("CNN Confusion Matrix:\n", conf_matrix_cnn)
# Calculate ROC AUC for CNN
fpr_cnn, tpr_cnn, _ = roc_curve(y_test, y_probs_cnn)
roc_auc_cnn = auc(fpr_cnn, tpr_cnn)
print("CNN ROC AUC:", roc_auc_cnn)

# Calculate and print accuracy, classification report, precision, recall, F1 score, and ROC AUC for LSTM
accuracy_lstm = accuracy_score(y_test, y_pred_lstm)
print("LSTM Accuracy:", accuracy_lstm)
print("LSTM Classification Report:\n", classification_report(y_test, y_pred_lstm))
print("LSTM Precision:", precision_score(y_test, y_pred_lstm, average='weighted'))
print("LSTM Recall:", recall_score(y_test, y_pred_lstm, average='weighted'))
print("LSTM F1 Score:", f1_score(y_test, y_pred_lstm, average='weighted'))
# Confusion Matrix for CNN
conf_matrix_lstm = confusion_matrix(y_test, y_pred_lstm)
print("LSTM Confusion Matrix:\n", conf_matrix_lstm)
# Calculate ROC AUC for CNN
fpr_lstm, tpr_lstm, _ = roc_curve(y_test, y_probs_lstm)
roc_auc_lstm = auc(fpr_lstm, tpr_lstm)
print("LSTM ROC AUC:", roc_auc_lstm)

# Calculate and print accuracy, classification report, precision, recall, F1 score, and ROC AUC for CNN
accuracy_hybrid = accuracy_score(y_test, y_pred_hybrid)
print("Hybrid Accuracy:", accuracy_hybrid)
print("Hybrid Classification Report:\n", classification_report(y_test, y_pred_hybrid))
print("Hybrid Precision:", precision_score(y_test, y_pred_hybrid, average='weighted'))
print("Hybrid Recall:", recall_score(y_test, y_pred_hybrid, average='weighted'))
print("Hybrid F1 Score:", f1_score(y_test, y_pred_hybrid, average='weighted'))
# Confusion Matrix for CNN
conf_matrix_hybrid = confusion_matrix(y_test, y_pred_hybrid)
print("Hybrid Confusion Matrix:\n", conf_matrix_cnn)
# Calculate ROC AUC for CNN
fpr_hybrid, tpr_hybrid, _ = roc_curve(y_test, y_probs_hybrid)
roc_auc_hybrid = auc(fpr_hybrid, tpr_hybrid)
print("HYBRID ROC AUC:", roc_auc_hybrid)

# Calculate ROC AUC for the GCN model
fpr_gcn, tpr_gcn, _ = roc_curve(y_test, y_probs_gcn)
roc_auc_gcn = auc(fpr_gcn, tpr_gcn)
print("GCN ROC AUC:", roc_auc_gcn)
# Calculate and print accuracy, classification report, precision, recall, F1 score, and ROC AUC for Transformer
y_pred_gcn = np.round(y_probs_gcn)
accuracy_gcn = accuracy_score(y_test, y_pred_gcn)
precision_gcn = precision_score(y_test, y_pred_gcn)
recall_gcn = recall_score(y_test, y_pred_gcn)
f1_gcn = f1_score(y_test, y_pred_gcn)
print("GCN Accuracy:", accuracy_gcn)
print("GCN Precision:", precision_gcn)
print("GCN Recall:", recall_gcn)
print("GCN F1 Score:", f1_gcn)

print(f'GAT-CL Test Accuracy: {accuracy:.4f}')
print(f'GAT-CL Test Accuracy: {accuracy_cl:.4f}')
print(f'GAT-CL F1 Score: {f1:.4f}')
print(f'GAT-CL Precision: {precision:.4f}')
print(f'GAT-CL Recall: {recall:.4f}')
print(f'GAT-CL AUC: {auc2:.4f}')
conf_matrix_gat_cl = confusion_matrix(y_test, predicted_labels_np)
print("GAT-CL Confusion Matrix:\n", conf_matrix_gat_cl)

fpr_gat, tpr_gat, _ = roc_curve(y_test_np, outputs_test_roc)
roc_auc_gat = auc(fpr_gat, tpr_gat)
print("GAT-CL ROC AUC:", roc_auc_gat)
# ROC Curve (only for binary classification)
#fprgat, tprgat, thresholds = roc_curve(y_test_np, outputs_test_roc)
#roc_auc_gat2 = auc(fprgat, tprgat)
# Plot combined ROC Curve
plt.figure(figsize=(12, 9))
plt.plot(fpr_svm, tpr_svm, color='lightblue', lw=3, label=f'SVM (AUC = {roc_auc_svm:.4f})')
plt.plot(fpr_cnn, tpr_cnn, color='red', lw=3, label=f'CNN (AUC = {roc_auc_cnn:.4f})')
plt.plot(fpr_lstm, tpr_lstm, color='green', lw=3, label=f'LSTM (AUC = {roc_auc_lstm:.4f})')
plt.plot(fpr_transformer, tpr_transformer, color='purple', lw=3, label=f'Transformer (AUC = {roc_auc_transformer:.4f})')
plt.plot(fpr_gcn, tpr_gcn, color='gray', lw=3, label=f'GCN (AUC = {roc_auc_gcn:.4f})')
plt.plot(fpr_gcn, tpr_gcn, color='darkgray', lw=3, label=f'HYBRID (AUC = {roc_auc_hybrid:.4f})')
plt.plot(fpr_gat, tpr_gat, color='darkblue', lw=3, label=f'GAT-CL (AUC = {roc_auc_gat:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=24)
plt.ylabel('True Positive Rate',fontsize=24)
plt.title('Lung Cancer Detection',fontsize=32)
plt.legend(loc="lower right", fontsize=18)
# Increase fontsize of X and Y axis values (tick labels)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.show()

tsne = TSNE(n_components=2, random_state=42)
tsne_representation = tsne.fit_transform(encoded_train_data)

# Plot the learned representations
plt.figure(figsize=(10, 8))
plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1], c=y_train, cmap='viridis', alpha=0.7)
plt.title('t-SNE Visualization of Learned Representations with GAT')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar()
plt.show()
