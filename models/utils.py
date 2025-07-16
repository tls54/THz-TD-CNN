
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

## Computer devide detection function
def identify_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    return device



# Load the validation dataset
def test_classifier(model, file_path, device):

    val_data = torch.load(file_path, weights_only=False)

    val_synthetic_data = val_data["synthetic_data"]         # shape: [N, 1024]
    val_num_layers = val_data["num_layers"]                 # shape: [N]
    val_num_layers_adjusted = val_num_layers - 1            # match training label indexing

    # Unsqueeze to match model input shape
    val_synthetic_data = val_synthetic_data.unsqueeze(1)    # shape: [N, 1, 1024]

    # Create DataLoader for validation
    val_dataset = TensorDataset(val_synthetic_data, val_num_layers_adjusted)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


    model.eval()
    all_preds = []
    all_labels = []

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())   # move back to CPU before numpy
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix and accuracy
    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)

    print(f"\n Unseen dataset Accuracy: {acc:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"{i+1}" for i in range(cm.shape[0])],
                yticklabels=[f"{i+1}" for i in range(cm.shape[0])])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Validation Set)")
    plt.show()