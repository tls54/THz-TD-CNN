import torch 
from torchsummary import summary


## Computer device detection function
def identify_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def display_model(summary_model, device):
    if device.type == "cuda":
        summary_model = summary_model.to(device)
        summary(summary_model, input_size=(1, 1024))
    elif device.type == "mps":
        summary_model = summary_model  # no .to(device) for MPS
        summary(summary_model, input_size=(1, 1024))
    else:
        summary_model = summary_model.to(device)
        summary(summary_model, input_size=(1, 1024))