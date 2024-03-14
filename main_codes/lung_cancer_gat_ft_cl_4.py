import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, confusion_matrix, \
    accuracy_score, RocCurveDisplay
from torch.optim.lr_scheduler import StepLR
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

data = pd.read_csv(r'..\dataset\feature_dataset.csv')
data.head()

y = data['CLASSES']
X = data.drop(columns=['CLASSES', 'Segment', 'Stage', 'PAT_ID'], axis=1)
#'CLASSES', 'Filename', 'Segment','MIN','MAX','MEAN','RMS','VAR','STD','POWER','PEAK','P2P','CREST FACTOR','SKEW','KURTOSIS','MAX_f','SUM_f','MEAN_f','VAR_f','PEAK_f','SKEW_f','KURTOSIS_f','PCA1','PCA2','PCA3'
X['KURTOSIS'] = pd.to_numeric(X['KURTOSIS'], errors='coerce')
X['KURTOSIS_f'] = pd.to_numeric(X['KURTOSIS_f'], errors='coerce')
X['SKEW'] = pd.to_numeric(X['SKEW'], errors='coerce')
X['SKEW_f'] = pd.to_numeric(X['SKEW_f'], errors='coerce')

# Feature transformation and selection (using FeatureWiz)
features = FeatureWiz(corr_limit=0.95, feature_engg='', category_encoders='', dask_xgboost_flag=False, nrows=None,
                      verbose=2)
X = features.fit_transform(X, y)
print(X.dtypes)
# Drop rows with NaN values (optional, depending on your data and goals)
X = X.dropna()
# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Assuming X_scaled.shape[0] is the number of nodes in your graph
num_nodes = X_scaled.shape[0]
# Create edge_index using torch_geometric.utils.add_self_loops
edge_index = add_self_loops(torch.zeros(2, num_nodes, dtype=torch.long), num_nodes=num_nodes)[0]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

import torch
import torch.nn.functional as F
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
    #CustomJitter(),  # Custom transformation for jitter
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
gat_hidden_size = 64      #64
gat_heads = 2            #2
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
# Contrastive loss fonksiyonunu tanımla
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

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
output_size_classifier = 2  # assuming 2 classes
classifier_model = DownstreamClassifier(input_size=output_size, output_size=output_size_classifier)
criterion_classifier = nn.CrossEntropyLoss()
optimizer_classifier = optim.AdamW(classifier_model.parameters(), lr=0.001)

# Train the downstream classifier
num_epochs_classifier = 3000
for epoch in range(num_epochs_classifier):
    optimizer_classifier.zero_grad()
    # Forward pass with softmax activation
    outputs_classifier = F.softmax(classifier_model(X_train_tensor), dim=1)
    # Flatten the y_train_tensor to 1D tensor
    y_train_tensor_flat = y_train_tensor.view(-1)
    # Use flattened y_train_tensor
    loss_classifier = criterion_classifier(outputs_classifier, y_train_tensor_flat.long())
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
    outputs_classifier = F.softmax(classifier_model(X_train_tensor), dim=1)
    # Flatten the y_train_tensor to 1D tensor
    y_train_tensor_flat = y_train_tensor.view(-1)
    # Assuming you have a binary classification problem
    # If you have more classes, adjust accordingly
    loss_classifier = criterion_classifier(outputs_classifier, y_train_tensor_flat.long())
    loss_classifier.backward()
    optimizer_fine_tune.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs_fine_tune}], Fine-tuned Classifier Loss: {loss_classifier.item():.4f}')

# Evaluate the fine-tuned classifier on the test set
with torch.no_grad():
    classifier_model.eval()
    outputs_test = classifier_model(X_test_tensor)[:,1]
    outputs_test2 = classifier_model(X_test_tensor)
    _, predicted_labels = torch.max(outputs_test2, 1)
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
    auc2 = roc_auc_score(y_test_np, predicted_labels_np, average='weighted')

    encoded_train_data = contrastive_model_with_gat_and_transformer.encoder(X_train_tensor).detach().numpy()

    conf_matrix_gat_cl = confusion_matrix(y_test, predicted_labels_np)
    print("SVM Confusion Matrix:\n", conf_matrix_gat_cl)

    print(f'Test Accuracy after Fine-tuning: {accuracy:.4f}')
    print(f'F1 Score after Fine-tuning: {f1:.4f}')
    print(f'Precision after Fine-tuning: {precision:.4f}')
    print(f'Recall after Fine-tuning: {recall:.4f}')
    print(f'AUC after Fine-tuning: {auc2:.4f}')
    conf_matrix_gat_cl = confusion_matrix(y_test, predicted_labels_np)
    print("GAT-CL Confusion Matrix:\n", conf_matrix_gat_cl)
    fpr_fine_tuned, tpr_fine_tuned, _ = roc_curve(y_test_np.ravel(), outputs_test.ravel())
    roc_auc_fine_tuned = auc(fpr_fine_tuned, tpr_fine_tuned)
    #fpr_fine_tuned["micro"], tpr_fine_tuned["micro"], _ = roc_curve(y_test_np.ravel(), predicted_labels_np.ravel())
    #roc_auc_fine_tuned["micro"] = auc(fpr_fine_tuned["micro"], tpr_fine_tuned["micro"])

    print("ROC AUC after Fine-tuning:", roc_auc_fine_tuned)

    # ROC Curve (only for binary classification)
    #fpr, tpr, thresholds = roc_curve(y_test_np, predicted_labels_np)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_fine_tuned, tpr_fine_tuned, label=f'AUC = {roc_auc_fine_tuned:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve after Fine-tuning')
    plt.legend()
    plt.show()