import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import os

# create output directory
output_dir = "/Users/soroush/Desktop/refs/figs"
os.makedirs(output_dir, exist_ok=True)

# we first define the oval distribution
OVAL_MU = 2.0
OVAL_SIGMA = 0.35
OVAL_COV_MATRIX = torch.tensor([[1.0, -0.71], [-0.71, 2.0]], dtype=torch.float32)
OVAL_CHOL = torch.linalg.cholesky(OVAL_COV_MATRIX)

def sampler_oval(n_samples, device='cpu'):
    theta = torch.rand(n_samples, device=device) * 2 * torch.pi
    r = torch.randn(n_samples, device=device) * OVAL_SIGMA + OVAL_MU
    y = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    x_samples = y @ OVAL_CHOL.T
    return x_samples

def pdf_oval(x_tensor):
    # If x = y @ L^T, then y = x @ (L^T)^(-1)
    OVAL_INV_CHOL_T = torch.linalg.inv(OVAL_CHOL.T) 
    y_transformed = x_tensor @ OVAL_INV_CHOL_T
    r = torch.linalg.norm(y_transformed, dim=1)
    dist_r = torch.distributions.Normal(loc=OVAL_MU, scale=OVAL_SIGMA)
    f_r = torch.exp(dist_r.log_prob(r))
    f_theta = 1 / (2 * torch.pi)
    jacobian = r * torch.abs(torch.linalg.det(OVAL_CHOL))
    pdf = f_r * f_theta / (jacobian + 1e-9)
    return pdf

def log_pdf_oval(x_tensor):
    return torch.log(pdf_oval(x_tensor) + 1e-9)

# compute maximum pairwise distance in the dataset to initialize largest noise lvel
def compute_max_pairwise_distance(n_samples=1000):
    data = sampler_oval(n_samples)
    # Compute pairwise distances
    dists = torch.cdist(data, data)
    max_dist = dists.max().item()
    return max_dist

# compute true score at different noise levels
def compute_true_score_noisy(x_tensor, sigma, n_samples_mc=1000):
    batch_size = x_tensor.shape[0]
    x_tensor = x_tensor.requires_grad_(True)
    
    # sample from the clean distribution
    y_samples = sampler_oval(n_samples_mc)
    
    scores = torch.zeros_like(x_tensor)
    
    for i in range(batch_size):
        x_i = x_tensor[i:i+1]
        
        # compute N(x_i; y, sigma2*I) for all y samples
        diff = x_i - y_samples
        log_gaussian = -0.5 * torch.sum(diff**2, dim=1) / (sigma**2) - torch.log(2 * np.pi * sigma**2)
        weights = torch.exp(log_gaussian)
        
        # normalize weights
        weights = weights / (weights.sum() + 1e-10)
        
        # compute weighted average of -(x-y)/sigma2
        score_contributions = -diff / (sigma**2)
        scores[i] = torch.sum(weights.unsqueeze(1) * score_contributions, dim=0)
    
    return scores.detach()

class ImprovedNCSN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=1024):
        super().__init__()
        
        self.network = nn.Sequential(
            # note the nput must be input_dim + 1 to account for log(sigma)
            nn.Linear(input_dim + 1, hidden_dim), 
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            # output dim same as input dim
            nn.Linear(hidden_dim, input_dim) 
        )
        
    def forward(self, x, sigma):
        if sigma.dim() == 0:
            sigma = sigma.expand(x.shape[0])
        elif sigma.dim() == 2:
            sigma = sigma.squeeze(1)
            
        log_sigma = torch.log(sigma).unsqueeze(1)
        x_sigma = torch.cat([x, log_sigma], dim=1)
        
        output = self.network(x_sigma)
        return output

# visualization of true vs learned scores at final stage
def visualize_final_scores(model, sigmas, grid_range=(-4, 4), n_points=40):
    print("visualizing scores at final stage")
    
    sigma_final = sigmas[-1]
    print(f"final sigma = {sigma_final:.4f}")
    
    x = np.linspace(grid_range[0], grid_range[1], n_points)
    y = np.linspace(grid_range[0], grid_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=-1)
    XY_tensor = torch.tensor(XY, dtype=torch.float32)
    
    # true pdf for heatmap
    pdf_true_vals = pdf_oval(XY_tensor).detach().numpy().reshape(X.shape)
    
    # true scores at lowest noise
    print("computing true scores at lowest noise level")
    if sigma_final < 0.05:
        # For very small noise, use clean score
        x_tensor_grad = XY_tensor.clone().requires_grad_(True)
        log_p = log_pdf_oval(x_tensor_grad)
        log_p.sum().backward()
        true_scores = x_tensor_grad.grad.clone().detach()
    else:
        true_scores = compute_true_score_noisy(XY_tensor, sigma_final, n_samples_mc=50000)
    
    U_true = true_scores[:, 0].reshape(X.shape)
    V_true = true_scores[:, 1].reshape(X.shape)
    
    # learned scores
    model.eval()
    with torch.no_grad():
        sigma_tensor = torch.tensor(sigma_final, dtype=torch.float32)
        learned_scores = model(XY_tensor, sigma_tensor)
    U_learned = learned_scores[:, 0].reshape(X.shape)
    V_learned = learned_scores[:, 1].reshape(X.shape)
    
    # calculate magnitudes for visualization
    mag_true = np.sqrt(U_true**2 + V_true**2)
    mag_true_flat = mag_true.flatten()
    mag_true_flat = mag_true_flat[np.isfinite(mag_true_flat)]
    scale_factor = np.median(mag_true_flat) * 5
    
    # density plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.contourf(X, Y, pdf_true_vals, levels=50, cmap='viridis')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ncsn_final_data_density.pdf', bbox_inches='tight')
    plt.close()
    
    # true score plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.contourf(X, Y, pdf_true_vals, levels=50, cmap='viridis', alpha=0.6)
    ax.quiver(X, Y, U_true, V_true, color='black', scale=scale_factor, 
              scale_units='xy', angles='xy', headwidth=5, width=0.003)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ncsn_final_true_scores.pdf', bbox_inches='tight')
    plt.close()
    
    # learned score plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.contourf(X, Y, pdf_true_vals, levels=50, cmap='viridis', alpha=0.6)
    ax.quiver(X, Y, U_learned, V_learned, color='black', scale=scale_factor, 
              scale_units='xy', angles='xy', headwidth=5, width=0.003)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ncsn_final_learned_scores.pdf', bbox_inches='tight')
    plt.close()
    
    # error analysis
    print("\n score matching quality at final stage:")
    
    # define regions, pdf based
    high_density_mask = pdf_true_vals > np.percentile(pdf_true_vals[pdf_true_vals > 1e-10], 70)
    low_density_mask = (pdf_true_vals > 1e-10) & (~high_density_mask)
    zero_density_mask = pdf_true_vals <= 1e-10
    
    # error metrics
    error_vec = true_scores.numpy() - learned_scores.numpy()
    error_magnitude = np.linalg.norm(error_vec, axis=1).reshape(X.shape)
    
    # angular error (in radians)
    true_norm = np.linalg.norm(true_scores.numpy(), axis=1, keepdims=True) + 1e-10
    learned_norm = np.linalg.norm(learned_scores.numpy(), axis=1, keepdims=True) + 1e-10
    true_unit = true_scores.numpy() / true_norm
    learned_unit = learned_scores.numpy() / learned_norm
    cos_angle = np.clip(np.sum(true_unit * learned_unit, axis=1), -1, 1)
    angular_error = np.arccos(cos_angle).reshape(X.shape)

# get noise schedule
def get_noise_schedule(n_sigmas=25):
    sigma_max = compute_max_pairwise_distance(5000)
    sigma_min = 0.01
    sigmas = torch.exp(torch.linspace(np.log(sigma_max), np.log(sigma_min), n_sigmas))
    print(f"noise schedule: sigma_max={sigma_max:.3f}, sigma_min={sigma_min:.3f}")
    return sigmas

# training with score matching loss

# helper to apply noise to batch
def collate_fn_add_noise(batch, sigmas):
    # but batch is list of tensors, so we stack them
    x = torch.stack([item[0] for item in batch])
    
    # select a random sigma for each sample in the batch
    labels = torch.randint(0, len(sigmas), (x.shape[0],))
    used_sigmas = sigmas[labels].view(x.shape[0], 1)
    
    # add noise
    noise = torch.randn_like(x)
    x_perturbed = x + used_sigmas * noise
    
    return x_perturbed, used_sigmas, noise

# training function
def train_ncsn_standard(num_epochs=51, batch_size=256, lr=1e-4):
    print("training NCSN with torch loop......bipbip")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    
    sigmas = get_noise_schedule(n_sigmas=20).to(device)
    model = ImprovedNCSN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # we first create a large and fixed dataset
    pool_size = 50000
    print(f"generating a fixed pool of {pool_size:,} samples")
    fixed_pool = sampler_oval(pool_size)
    dataset = TensorDataset(fixed_pool)
    
    # then create the DataLoader with our helper
    # use a lambda to pass the sigmas tensor to the helper
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda batch: collate_fn_add_noise(batch, sigmas)
    )
    
    losses = []
    
    # training loop
    for epoch in range(num_epochs):
        model.train()
        avg_loss = 0.0
        for x_perturbed, used_sigmas, noise in data_loader:
            x_perturbed = x_perturbed.to(device)
            used_sigmas = used_sigmas.to(device)
            noise = noise.to(device)
            
            optimizer.zero_grad()
            
            scores = model(x_perturbed, used_sigmas.squeeze())
            
            target = -noise
            loss = 0.5 * torch.mean(torch.sum((used_sigmas * scores - target)**2, dim=1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            avg_loss += loss.item()
            
        scheduler.step()
        avg_loss /= len(data_loader)
        losses.append(avg_loss)
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"epoch [{epoch+1}/{num_epochs}], Avg loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
            
    print("training complete!!!!!!")
    return model, losses, sigmas

# annealed LD
def annealed_langevin_dynamics(model, sigmas, num_samples=500, epsilon=1e-4, T=200):
    model.eval()
    
    x = torch.randn(num_samples, 2) * sigmas[0]
    trajectory = [x.clone().numpy()]
    
    for i, sigma in enumerate(sigmas):
        alpha_i = epsilon * (sigma / sigmas[-1]) ** 2
        if i >= len(sigmas) - 3:
            alpha_i = alpha_i * 0.5

        for t in range(T):
            z = torch.randn_like(x)
            
            with torch.no_grad():
                score = model(x, sigma)
            
            x = x + (alpha_i / 2) * score + torch.sqrt(torch.tensor(alpha_i)) * z
        
        trajectory.append(x.clone().numpy())
        if (i + 1) % 5 == 0:
            print(f"completed noise level {i+1}/{len(sigmas)}, sigma={sigma:.3f}")
    
    return np.array(trajectory), x.numpy()

# compute noise-perturbed density p_sigma(x) = int [p_0(y) N(x; y, sigma2*I] dy with MC
def compute_perturbed_density(x_grid, sigma, n_samples=5000):
    # x_grid no require gradients
    x_grid = x_grid.detach()
    
    # sample from the clean distribution
    y_samples = sampler_oval(n_samples)
    
    # for each point in the grid, compute the density
    density = torch.zeros(x_grid.shape[0])
    
    for i in range(x_grid.shape[0]):
        x_i = x_grid[i:i+1]
        # compute N(x_i; y, sigma2*I) for all y samples
        diff = x_i - y_samples
        log_gaussian = -0.5 * torch.sum(diff**2, dim=1) / (sigma**2) - torch.log(2 * np.pi * sigma**2)
        # average the gaussian densities
        density[i] = torch.exp(log_gaussian).mean()
    
    return density.numpy()

# visualize perturbed density and scores at multiple noise levels
def visualize_scores_multi_noise(model, sigmas, grid_range=(-4, 4), n_points=40):    
    # three noise levels: high, medium, low
    indices = [0, len(sigmas)//4, -1]
    selected_sigmas = [sigmas[idx] for idx in indices]
    noise_labels = ['high_noise', 'medium_noise', 'low_noise']
    
    # grid for evaluation
    x = np.linspace(grid_range[0], grid_range[1], n_points)
    y = np.linspace(grid_range[0], grid_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=-1)
    XY_tensor = torch.tensor(XY, dtype=torch.float32)
    
    # pdf for reference
    pdf_clean = pdf_oval(XY_tensor).detach().numpy().reshape(X.shape)
    
    for noise_idx, (sigma_idx, sigma, noise_label) in enumerate(zip(indices, selected_sigmas, noise_labels)):
        print(f"processing {noise_label}: sigma = {sigma:.3f}")
        
        # Compute perturbed density
        print(f"computing perturbed density")
        if sigma < 0.05:
            print("clean pdf only for visualization")
            perturbed_density = pdf_oval(XY_tensor).detach().numpy()
        else:
            print("perturbed density with MC")
            perturbed_density = compute_perturbed_density(XY_tensor, sigma, n_samples=50000)
        
        perturbed_density = perturbed_density.reshape(X.shape)
        
        # compute true scores at this noise level
        print(f"computing true scores")
        if sigma > 2.0:
            # For large noise, use gaussian approximation
            true_scores = -XY_tensor / (sigma**2)
        else:
            # for small noise, check if we should use clean scores
            if sigma < 0.05:
                # using clean scores for very low noise
                x_tensor_grad = XY_tensor.clone().requires_grad_(True)
                x_tensor_grad.retain_grad()
                log_p = log_pdf_oval(x_tensor_grad)
                log_p.sum().backward()
                true_scores = x_tensor_grad.grad.clone().detach()
            else:
                true_scores = compute_true_score_noisy(XY_tensor, sigma, n_samples_mc=100000)
        
        U_true = true_scores[:, 0].reshape(X.shape)
        V_true = true_scores[:, 1].reshape(X.shape)
        
        # compute learned scores
        model.eval()
        with torch.no_grad():
            sigma_tensor = torch.tensor(sigma, dtype=torch.float32)
            learned_scores = model(XY_tensor, sigma_tensor)
        U_learned = learned_scores[:, 0].reshape(X.shape)
        V_learned = learned_scores[:, 1].reshape(X.shape)
        
        # calculate scale for vectors
        mag_true = np.sqrt(U_true**2 + V_true**2)
        scale_factor = np.median(mag_true[mag_true > 0]) * 5
        
        # perturbed density plot
        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.contourf(X, Y, perturbed_density, levels=50, cmap='hot')
        ax.set_xlim(grid_range)
        ax.set_ylim(grid_range)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ncsn_perturbed_density_{noise_label}.pdf', bbox_inches='tight')
        plt.close()
        
        #true scores on perturbed density plot
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.contourf(X, Y, perturbed_density, levels=50, cmap='Blues_r', alpha=0.7)
        skip = 3
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                  U_true[::skip, ::skip], V_true[::skip, ::skip], 
                  color='darkblue', scale=scale_factor, scale_units='xy', 
                  angles='xy', headwidth=4, width=0.004)
        ax.set_xlim(grid_range)
        ax.set_ylim(grid_range)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ncsn_true_scores_{noise_label}.pdf', bbox_inches='tight')
        plt.close()
        
        #learned scores on perturbed density plot
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.contourf(X, Y, perturbed_density, levels=50, cmap='Blues_r', alpha=0.7)
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                  U_learned[::skip, ::skip], V_learned[::skip, ::skip], 
                  color='darkblue', scale=scale_factor, scale_units='xy', 
                  angles='xy', headwidth=4, width=0.004)
        ax.set_xlim(grid_range)
        ax.set_ylim(grid_range)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ncsn_learned_scores_{noise_label}.pdf', bbox_inches='tight')
        plt.close()

# to show error magnitude and angular error at multiple noise levels.
def visualize_error_analysis_multi_noise(model, sigmas, grid_range=(-4, 4), n_points=100):
    # three noise levels: high, medium, low
    indices = [0, len(sigmas)//2, -1]
    selected_sigmas = [sigmas[idx] for idx in indices]
    noise_labels = ['high_noise', 'medium_noise', 'low_noise']
    
    # grid for evaluation
    x = np.linspace(grid_range[0], grid_range[1], n_points)
    y = np.linspace(grid_range[0], grid_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=-1)
    XY_tensor = torch.tensor(XY, dtype=torch.float32)
    
    # true PDF for density contours
    pdf_clean = pdf_oval(XY_tensor).detach().numpy().reshape(X.shape)
    
    for noise_idx, (sigma_idx, sigma, noise_label) in enumerate(zip(indices, selected_sigmas, noise_labels)):
        print(f"processing {noise_label}: sigma = {sigma:.3f}")
        
        # compute true scores at this noise level
        if sigma > 2.0:
            # for large noise, use gaussian approximation
            true_scores = -XY_tensor / (sigma**2)
        else:
            # for small noise check if we should use clean scores
            if sigma < 0.05:
                # use clean scores for very low noise
                x_tensor_grad = XY_tensor.clone().requires_grad_(True)
                x_tensor_grad.retain_grad()
                log_p = log_pdf_oval(x_tensor_grad)
                log_p.sum().backward()
                true_scores = x_tensor_grad.grad.clone().detach()
            else:
                true_scores = compute_true_score_noisy(XY_tensor, sigma, n_samples_mc=100000)
        
        # learned scores
        model.eval()
        with torch.no_grad():
            sigma_tensor = torch.full((XY_tensor.shape[0],), sigma, dtype=torch.float32)
            learned_scores = model(XY_tensor, sigma_tensor)
        
        # error metrics
        error_vec = true_scores.numpy() - learned_scores.numpy()
        error_magnitude = np.linalg.norm(error_vec, axis=1).reshape(X.shape)
        
        # angular error
        true_norm = np.linalg.norm(true_scores.numpy(), axis=1, keepdims=True) + 1e-10
        learned_norm = np.linalg.norm(learned_scores.numpy(), axis=1, keepdims=True) + 1e-10
        true_unit = true_scores.numpy() / true_norm
        learned_unit = learned_scores.numpy() / learned_norm
        cos_angle = np.clip(np.sum(true_unit * learned_unit, axis=1), -1, 1)
        angular_error = np.arccos(cos_angle).reshape(X.shape) * 180 / np.pi
        
        #rror magnitude heatmap plot
        fig, ax = plt.subplots(figsize=(7, 7))
        
        # change scale arrows too big or small
        vmax = np.percentile(error_magnitude, 95)  # 95th percentile to avoid outliers
        
        im1 = ax.contourf(X, Y, error_magnitude, levels=50, cmap='hot', vmin=0, vmax=100)
        ax.contour(X, Y, pdf_clean, levels=[0.01*np.max(pdf_clean), 0.1*np.max(pdf_clean)],
                   colors='cyan', linewidths=2, alpha=0.7)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_aspect('equal')
        plt.colorbar(im1, ax=ax)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ncsn_error_magnitude_{noise_label}.pdf', bbox_inches='tight')
        plt.close()
        
        #a ngular erorr heatmap
        fig, ax = plt.subplots(figsize=(7, 7))
        
        # cap it @ 180 degrees
        angular_error = np.minimum(angular_error, 180)
        
        im2 = ax.contourf(X, Y, angular_error, levels=50, cmap='plasma', vmin=0, vmax=180)
        ax.contour(X, Y, pdf_clean, levels=[0.01*np.max(pdf_clean), 0.1*np.max(pdf_clean)],
                   colors='cyan', linewidths=2, alpha=0.7)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_aspect('equal')
        plt.colorbar(im2, ax=ax, label='degrees')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ncsn_angular_error_{noise_label}.pdf', bbox_inches='tight')
        plt.close()
        
        # statistics
        print(f"\n{noise_label} Error Statistics:")
        print(f"  Mean Error Magnitude: {np.mean(error_magnitude):.4f}")
        print(f"  Max Error Magnitude: {np.max(error_magnitude):.4f}")
        print(f"  Mean Angular Error: {np.mean(angular_error):.1f}deg")
        print(f"  Max Angular Error: {np.max(angular_error):.1f}deg")
        
        # more statistics
        high_density_mask = pdf_clean > np.percentile(pdf_clean[pdf_clean > 1e-10], 70)
        low_density_mask = (pdf_clean > 1e-10) & (~high_density_mask)
        zero_density_mask = pdf_clean <= 1e-10
        
        print(f"  High Density Region MAE: {np.mean(error_magnitude[high_density_mask]):.4f}")
        print(f"  Low Density Region MAE: {np.mean(error_magnitude[low_density_mask]):.4f}")
        print(f"  Zero Density Region MAE: {np.mean(error_magnitude[zero_density_mask]):.4f}")

#  main
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    
    # train model
    model, losses, sigmas = train_ncsn_standard(num_epochs=51)
    
    # plot training loss
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    epochs = range(1, len(losses) + 1)  # Start from epoch 1
    ax.plot(epochs, losses, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ncsn_training_loss.pdf', bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close()

    # error analysis at multiple noise levels
    visualize_error_analysis_multi_noise(model, sigmas)

    # bisualize scores at multiple noise levels
    visualize_scores_multi_noise(model, sigmas)
    
    # visualize true vs learned scores at final stage
    visualize_final_scores(model, sigmas)
    
    # generate samples
    trajectory, final_samples = annealed_langevin_dynamics(
        model, sigmas, num_samples=500, epsilon=1e-4, T=200
    )
    
    # sample evolution visualization - 3 stages of annealing    
    # true distribution for contours
    x_grid = np.linspace(-6, 6, 100)
    y_grid = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    XY_tensor = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=-1), dtype=torch.float32)
    pdf_vals = pdf_oval(XY_tensor).numpy().reshape(X.shape)
    
    # start, middle, end
    stage_indices = [0, len(trajectory)//3, -1]  # First, middle, last
    stage_names = ['initial_high_noise', 'middle_noise', 'final_low_noise']
    stage_labels = ['High Noise (Initial)', 'Medium Noise (Middle)', 'Low Noise (Final)']
    
    for stage_idx, (traj_idx, stage_name, stage_label) in enumerate(zip(stage_indices, stage_names, stage_labels)):
        samples_at_stage = trajectory[traj_idx]
        
        fig, ax = plt.subplots(figsize=(7, 7))
        
        # filled contours (true distribution)
        ax.contourf(X, Y, pdf_vals, levels=50, cmap='viridis', alpha=0.6)
        
        # scatter plot of samples
        ax.scatter(samples_at_stage[:, 0], samples_at_stage[:, 1], 
                  s=8, color='black', edgecolors='black', linewidths=0.3)
        
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')  
        ax.set_aspect('equal')
        
        # add grid for better visualization
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ncsn_sample_evolution_{stage_name}.pdf', 
                   bbox_inches='tight')
        plt.close()