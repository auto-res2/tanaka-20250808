import os
import torch
from src.preprocess import get_dataloader
from src.train import NetQ, NetG, NetD, IALWGAN, train_model
from src.evaluate import experiment_2, experiment_3
import matplotlib.pyplot as plt


def experiment_1(device, dataloader):
    print("\nStarting Experiment 1: Ablation Study on the Isometric Regularizer")
    z_dim = 100
    netQ = NetQ(z_dim)
    netG = NetG(z_dim)
    netD = NetD()
    
    # Full model with isometric loss
    full_model = IALWGAN(z_dim, netQ, netG, netD, device=device)
    # Baseline model without isometric loss (we use the same architecture and disable iso regularization via lambda_iso=0)
    baseline_model = IALWGAN(z_dim, netQ, netG, netD, device=device)
    
    import torch.optim as optim
    optimizer_full = optim.Adam(full_model.parameters(), lr=0.0002)
    optimizer_baseline = optim.Adam(baseline_model.parameters(), lr=0.0002)
    
    print("Training Full Model with isometric regularizer (λ_iso = 1.0)")
    loss_history_full = train_model(full_model, dataloader, optimizer_full, num_epochs=5, lambda_iso=1.0)
    
    print("Training Baseline Model without isometric regularizer (λ_iso = 0.0)")
    loss_history_baseline = train_model(baseline_model, dataloader, optimizer_baseline, num_epochs=5, lambda_iso=0.0)
    
    os.makedirs('./.research/iteration1/images', exist_ok=True)
    plt.figure()
    plt.plot(loss_history_full, label="Full Model")
    plt.plot(loss_history_baseline, label="Baseline")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.title("Loss Curve Comparison for Ablation Study")
    filename = './.research/iteration1/images/training_loss_ablation.pdf'
    plt.savefig(filename, bbox_inches="tight")
    print(f"Saved loss curve plot as {filename}")
    plt.close()


def test_code():
    print("\n===== Starting Test Run =====")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    dataloader = get_dataloader(batch_size=64, train=True)
    
    experiment_1(device, dataloader)
    experiment_2(device, dataloader)
    experiment_3(device, dataloader)
    
    print("Test run finished immediately. If you see PDF files generated in .research/iteration1/images and printed outputs, the code is working.")


if __name__ == '__main__':
    test_code()
