import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils
from scipy.stats import pearsonr

# Import the networks and training function
from src.train import NetQ, NetG, NetD, IALWGAN, train_model


def evaluate_lambda(lambda_iso_value, model_constructor, dataloader, device):
    """ Train one instance of IALWGAN with a given lambda_iso value and return metrics."""
    z_dim = 100
    netQ = NetQ(z_dim)
    netG = NetG(z_dim)
    netD = NetD()
    model = model_constructor(z_dim, netQ, netG, netD, device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    print(f"Training with λ_iso = {lambda_iso_value}")
    loss_history = train_model(model, dataloader, optimizer, num_epochs=3, lambda_iso=lambda_iso_value)
    
    model.eval()
    recon_error = 0.0
    mse_loss = nn.MSELoss(reduction='sum')
    total_samples = 0
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            rec = model.netG(model.netQ(data, rank=0))
            recon_error += mse_loss(data, rec).item()
            total_samples += data.size(0)
            break  # quick demo with one batch
    recon_error /= total_samples
    
    batch_data, _ = next(iter(dataloader))
    batch_data = batch_data.to(device)
    with torch.no_grad():
        latent_vectors = model.netQ(batch_data, rank=0)
    batch_flat = batch_data.view(batch_data.size(0), -1)
    latent_dists = torch.cdist(latent_vectors, latent_vectors, p=2).cpu().numpy().flatten()
    data_dists = torch.cdist(batch_flat, batch_flat, p=2).cpu().numpy().flatten()
    r, _ = pearsonr(latent_dists, data_dists)
    
    return loss_history, recon_error, r


def experiment_2(device, dataloader):
    print("\nStarting Experiment 2: Hyperparameter Sensitivity Analysis for Isometric Loss Weight (λ₂)")
    lambda_values = [0.0, 0.1, 0.5, 1.0, 2.0]
    results = {}
    for lam in lambda_values:
        history, mse, corr = evaluate_lambda(lam, IALWGAN, dataloader, device)
        results[lam] = {'loss_history': history, 'recon_error': mse, 'distance_corr': corr}
        print(f"λ_iso: {lam}, Reconstruction Error: {mse:.4f}, Distance Correlation: {corr:.4f}")

    lambdas = list(results.keys())
    recon_errors = [results[l]['recon_error'] for l in lambdas]
    correlations = [results[l]['distance_corr'] for l in lambdas]
    
    os.makedirs('./.research/iteration1/images', exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(lambdas, recon_errors, marker='o')
    plt.xlabel("λ₂ (isometric loss weight)")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.title("Reconstruction Error vs. λ₂")
    
    plt.subplot(1, 2, 2)
    plt.plot(lambdas, correlations, marker='x', color='red')
    plt.xlabel("λ₂ (isometric loss weight)")
    plt.ylabel("Latent-Data Distance Correlation")
    plt.title("Distance Correlation vs. λ₂")
    plt.tight_layout()
    filename = './.research/iteration1/images/loss_vs_lambda.pdf'
    plt.savefig(filename, bbox_inches="tight")
    print(f"Saved hyperparameter sensitivity plots as {filename}")
    plt.close()


def extract_embeddings(model, dataloader, num_samples=200):
    model.eval()
    embeddings = []
    images = []
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(model.device)
            emb = model.netQ(data, rank=0)
            embeddings.append(emb.cpu())
            images.append(data.cpu())
            if sum(item.size(0) for item in embeddings) >= num_samples:
                break
    embeddings = torch.cat(embeddings, dim=0)[:num_samples]
    images = torch.cat(images, dim=0)[:num_samples]
    return embeddings, images


def compute_distance_correlation(embeddings, feature_output=None):
    if feature_output is None:
        feature_output = embeddings
    latent_dists = torch.cdist(embeddings, embeddings, p=2).numpy().flatten()
    feature_dists = torch.cdist(feature_output, feature_output, p=2).numpy().flatten()
    r, _ = pearsonr(latent_dists, feature_dists)
    return r


def latent_interpolation(model, img1, img2, steps=10, device=torch.device('cpu')):
    model.eval()
    interpolated = []
    with torch.no_grad():
        z1 = model.netQ(img1.unsqueeze(0).to(device), rank=0)
        z2 = model.netQ(img2.unsqueeze(0).to(device), rank=0)
        for alpha in np.linspace(0, 1, steps):
            z_interp = (1 - alpha) * z1 + alpha * z2
            generated = model.netG(z_interp)
            interpolated.append(generated.squeeze(0).cpu())
    return interpolated


def plot_interpolations(interpolations, title="Interpolation", filename="interpolation.pdf"):
    grid = utils.make_grid(torch.stack(interpolations), nrow=len(interpolations), normalize=True, scale_each=True)
    plt.figure(figsize=(15, 5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(title)
    plt.axis("off")
    file_path = os.path.join('./.research/iteration1/images', filename)
    plt.savefig(file_path, bbox_inches="tight")
    print(f"Saved interpolation plot as {file_path}")
    plt.close()


def experiment_3(device, dataloader):
    print("\nStarting Experiment 3: Visual and Quantitative Evaluation of Latent Space Geometry")
    from src.train import NetQ, NetG, NetD, IALWGAN, train_model
    z_dim = 100
    # Train full model
    netQ_full = NetQ(z_dim)
    netG_full = NetG(z_dim)
    netD_full = NetD()
    full_model = IALWGAN(z_dim, netQ_full, netG_full, netD_full, device=device)
    optimizer_full = optim.Adam(full_model.parameters(), lr=0.0002)
    train_model(full_model, dataloader, optimizer_full, num_epochs=3, lambda_iso=1.0)
    
    # Train baseline model without isometric loss
    netQ_base = NetQ(z_dim)
    netG_base = NetG(z_dim)
    netD_base = NetD()
    baseline_model = IALWGAN(z_dim, netQ_base, netG_base, netD_base, device=device)
    optimizer_base = optim.Adam(baseline_model.parameters(), lr=0.0002)
    train_model(baseline_model, dataloader, optimizer_base, num_epochs=3, lambda_iso=0.0)
    
    embeddings_full, _ = extract_embeddings(full_model, dataloader, num_samples=200)
    embeddings_base, _ = extract_embeddings(baseline_model, dataloader, num_samples=200)
    corr_full = compute_distance_correlation(embeddings_full)
    corr_base = compute_distance_correlation(embeddings_base)
    print(f"Latent-Distance Correlation (Full Model): {corr_full:.4f}")
    print(f"Latent-Distance Correlation (Baseline): {corr_base:.4f}")
    
    sample_imgs, _ = next(iter(dataloader))
    img1, img2 = sample_imgs[0], sample_imgs[1]
    interpolations_full = latent_interpolation(full_model, img1, img2, steps=10, device=device)
    interpolations_base = latent_interpolation(baseline_model, img1, img2, steps=10, device=device)
    
    plot_interpolations(interpolations_full, title="Full Model Latent Interpolations", filename="interpolation_full.pdf")
    plot_interpolations(interpolations_base, title="Baseline Model Latent Interpolations", filename="interpolation_baseline.pdf")


if __name__ == '__main__':
    print("This is the evaluation module. Import and run its functions from main.")
