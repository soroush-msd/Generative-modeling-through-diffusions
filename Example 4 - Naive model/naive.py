import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os
from torch.utils.data import TensorDataset, DataLoader

output_dir = "/Users/soroush/Desktop/refs/figs"  # change if needed
os.makedirs(output_dir, exist_ok=True)

# oval distribution
OVAL_MU = 2.0
OVAL_SIGMA = 0.35
OVAL_COV_MATRIX = torch.tensor([[1.0, -0.71], [-0.71, 2.0]], dtype=torch.float32)
OVAL_CHOL = torch.linalg.cholesky(OVAL_COV_MATRIX)
OVAL_INV_CHOL = torch.linalg.inv(OVAL_CHOL)
OVAL_INV_SQRT = OVAL_INV_CHOL.T
OVAL_DET_INV_SQRT = torch.linalg.det(OVAL_INV_SQRT)

# error analysis
def analyze_score_errors(model, true_score_fn, pdf_fn, grid_range=(-4, 4), n_points=200):

    # evaluation grid
    x = np.linspace(grid_range[0], grid_range[1], n_points)
    y = np.linspace(grid_range[0], grid_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=-1)
    XY_tensor = torch.tensor(XY, dtype=torch.float32)

    # true pdf
    pdf_vals = pdf_fn(XY_tensor).detach().numpy().reshape(X.shape)

    # true and leanred scores
    true_scores = true_score_fn(XY_tensor).detach().numpy()
    model.eval()
    with torch.no_grad():
        learned_scores = model(XY_tensor).numpy()

    # for visualization
    U_true = true_scores[:, 0].reshape(X.shape)
    V_true = true_scores[:, 1].reshape(X.shape)
    U_learned = learned_scores[:, 0].reshape(X.shape)
    V_learned = learned_scores[:, 1].reshape(X.shape)

    # error metrics
    error_vec = true_scores - learned_scores
    error_magnitude = np.linalg.norm(error_vec, axis=1).reshape(X.shape)

    # angular error (in radians)
    true_norm = np.linalg.norm(true_scores, axis=1, keepdims=True) + 1e-10
    learned_norm = np.linalg.norm(learned_scores, axis=1, keepdims=True) + 1e-10
    true_unit = true_scores / true_norm
    learned_unit = learned_scores / learned_norm
    cos_angle = np.clip(np.sum(true_unit * learned_unit, axis=1), -1, 1)
    angular_error = np.arccos(cos_angle).reshape(X.shape)

    # regions based on PDF values
    thresh = np.percentile(pdf_vals, 90)  # top 10% of all pdf samples
    high_density_mask = pdf_vals >= thresh
    low_density_mask = (pdf_vals > 1e-4) & (~high_density_mask)
    zero_density_mask = pdf_vals <= 1e-4

    # Save individual plots
    
    # rrror mag heatmap
    fig, ax = plt.subplots(figsize=(7, 7))
    im1 = ax.contourf(X, Y, error_magnitude, levels=50, cmap='hot')
    ax.contour(X, Y, pdf_vals, levels=[0.01*np.max(pdf_vals), 0.1*np.max(pdf_vals)], 
              colors='cyan', linewidths=2, alpha=0.7)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_aspect('equal')
    plt.colorbar(im1, ax=ax)
    plt.tight_layout()
    plt.savefig('/Users/soroush/Desktop/refs/figs/error_magnitude_heatmap.pdf', bbox_inches='tight')
    plt.close()

    # angular rrror heatmap
    fig, ax = plt.subplots(figsize=(7, 7))
    im2 = ax.contourf(X, Y, angular_error * 180/np.pi, levels=50, cmap='plasma')
    ax.contour(X, Y, pdf_vals, levels=[0.01*np.max(pdf_vals), 0.1*np.max(pdf_vals)], 
              colors='cyan', linewidths=2, alpha=0.7)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_aspect('equal')
    plt.colorbar(im2, ax=ax, label='degrees')
    plt.tight_layout()
    plt.savefig('/Users/soroush/Desktop/refs/figs/angular_error_heatmap.pdf', bbox_inches='tight')
    plt.close()

    # regional error stats
    fig, ax = plt.subplots(figsize=(7, 7))
    stats = {
        'High Density\n(Ring)': {
            'mask': high_density_mask,
            'color': 'green'
        },
        'Low Density\n(Transition)': {
            'mask': low_density_mask,
            'color': 'orange'
        },
        'Zero Density\n(Hole+Corners)': {
            'mask': zero_density_mask,
            'color': 'red'
        }
    }

    positions = []
    labels = []
    colors = []
    for i, (region, info) in enumerate(stats.items()):
        mask = info['mask']
        if np.any(mask):
            errors_in_region = error_magnitude[mask]
            positions.extend([i*3 + j for j in range(3)])
            labels.extend([region if j == 1 else '' for j in range(3)])
            colors.extend([info['color']] * 3)

            # Box plot data
            bp = ax.boxplot([errors_in_region], positions=[i*3+1], widths=0.8, 
                          patch_artist=True, showfliers=False)
            bp['boxes'][0].set_facecolor(info['color'])
            bp['boxes'][0].set_alpha(0.6)

            # Add mean as a point
            mean_error = np.mean(errors_in_region)
            ax.scatter([i*3+1], [mean_error], color='black', s=50, zorder=10)

    ax.set_xticks([1, 4, 7])
    ax.set_xticklabels(['High Density\n(Ring)', 'Low Density\n(Transition)', 'Zero Density\n(Hole+Corners)'])
    ax.set_ylabel('Error Magnitude')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/soroush/Desktop/refs/figs/regional_error_statistics.pdf', bbox_inches='tight')
    plt.close()

    # summary
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.axis('off')
    
    # overall stats
    overall_mse = np.mean(error_magnitude**2)
    overall_mae = np.mean(error_magnitude)
    overall_max = np.max(error_magnitude)

    # more stats to see
    summary_text = "error stat summary\n" + "="*30 + "\n\n"
    summary_text += f"overall MSE: {overall_mse:.4f}\n"
    summary_text += f"pverall MAE: {overall_mae:.4f}\n"
    summary_text += f"max error: {overall_max:.4f}\n\n"
    summary_text += "regional:\n" + "-"*20 + "\n"

    for region, info in stats.items():
        mask = info['mask']
        if np.any(mask):
            errors = error_magnitude[mask]
            region_clean = region.replace('\n', ' ')
            summary_text += f"\n{region_clean}:\n"
            summary_text += f" MAE: {np.mean(errors):.4f}\n"
            summary_text += f" Max: {np.max(errors):.4f}\n"
            summary_text += f" coverage: {100*np.sum(mask)/mask.size:.1f}%\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig('/Users/soroush/Desktop/refs/figs/error_statistics_summary.pdf', bbox_inches='tight')
    plt.close()

    return {
        'error_magnitude': error_magnitude,
        'angular_error': angular_error,
        'overall_mse': overall_mse,
        'overall_mae': overall_mae,
        'regional_masks': {
            'high_density': high_density_mask,
            'low_density': low_density_mask,
            'zero_density': zero_density_mask
        }
    }

def pdf_oval(x_tensor):
    OVAL_INV_CHOL_T = torch.linalg.inv(OVAL_CHOL.T)
    y_transformed = x_tensor @ OVAL_INV_CHOL_T
    r = torch.linalg.norm(y_transformed, dim=1)
    
    dist_r = torch.distributions.Normal(loc=OVAL_MU, scale=OVAL_SIGMA)
    f_r = torch.exp(dist_r.log_prob(r))
    f_theta = 1 / (2 * torch.pi)
    jacobian = r * torch.abs(torch.linalg.det(OVAL_CHOL))
    
    pdf = f_r * f_theta / (jacobian + 1e-9)
    return pdf
# llog pdf
def log_pdf_oval(x_tensor):
    # epsilon for stability
    return torch.log(pdf_oval(x_tensor) + 1e-9)

def sampler_oval(n_samples, device='cpu'):
    theta = torch.rand(n_samples, device=device) * 2 * torch.pi
    r = torch.randn(n_samples, device=device) * OVAL_SIGMA + OVAL_MU
    y = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    x_samples = y @ OVAL_CHOL.T 
    return x_samples

# true score via autograd
def true_score_oval(x_tensor):
    x_tensor_clone = x_tensor.clone().detach().requires_grad_(True)
    log_p_x = log_pdf_oval(x_tensor_clone)
    log_p_x.sum().backward()
    score = x_tensor_clone.grad.clone()
    return score

class ScoreNet2D(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=1024, output_dim=2):
        super(ScoreNet2D, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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
            nn.Linear(hidden_dim, output_dim)
        )

        # better init
        self.network[-1].weight.data.uniform_(-0.01, 0.01)

    def forward(self, x):
        return self.network(x)

# train score model with loader
def train_with_score_matching_oval(num_epochs=51, batch_size=256, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    model = ScoreNet2D().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # a large fixed pool
    pool_size = 50000
    print(f"generating a fixed pool of {pool_size:,} samples...")
    fixed_pool = sampler_oval(pool_size)

    # TensorDataset and DataLoader for automatic batching and shuffling
    dataset = TensorDataset(fixed_pool)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # sheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Store average loss per epoch (not per batch)
    epoch_losses = []

    # training loop
    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        
        for i, (x_train_batch,) in enumerate(train_loader):
            x_train = x_train_batch.to(device).requires_grad_(True)

            optimizer.zero_grad()

            s_pred = model(x_train)

            # score matching Loss
            term1 = 0.5 * torch.sum(s_pred**2, dim=1)
            tr_jacobian = torch.zeros(x_train.shape[0], device=device)

            for j in range(x_train.shape[1]):
                grad = torch.autograd.grad(s_pred[:, j].sum(), x_train, create_graph=True)[0]
                tr_jacobian += grad[:, j]

            loss = (tr_jacobian + term1).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_losses.append(loss.item())

        # average loss/epoich
        avg_epoch_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_epoch_loss)

        # move the scheduler at the end of each epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"epoch [{epoch+1}/{num_epochs}], Avg loss: {avg_epoch_loss:.4f}, LR: {current_lr:.6f}")

    print("training complete!!!!!")
    return model, epoch_losses

# cool visualizations
def visualize_scores_oval(model, grid_range=(-4, 4), n_points=30):
    x = np.linspace(grid_range[0], grid_range[1], n_points)
    y = np.linspace(grid_range[0], grid_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=-1)
    XY_tensor = torch.tensor(XY, dtype=torch.float32)

    # true pdf for heatmap
    pdf_true_vals = pdf_oval(XY_tensor).detach().numpy().reshape(X.shape)

    # true scores
    scores_true_vals = true_score_oval(XY_tensor).detach().numpy()
    U_true, V_true = scores_true_vals[:, 0].reshape(X.shape), scores_true_vals[:, 1].reshape(X.shape)

    # learned scores
    model.eval()
    with torch.no_grad():
        scores_learned_vals = model(XY_tensor).numpy()
    U_learned, V_learned = scores_learned_vals[:, 0].reshape(X.shape), scores_learned_vals[:, 1].reshape(X.shape)

    # magnitudes
    mag_true = np.sqrt(U_true**2 + V_true**2)
    mag_learned = np.sqrt(U_learned**2 + V_learned**2)

    # using median for more robust scaling
    mag_true_flat = mag_true.flatten()
    mag_true_flat = mag_true_flat[np.isfinite(mag_true_flat)]
    scale_factor = np.median(mag_true_flat) * 10

    # data density plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.contourf(X, Y, pdf_true_vals, levels=50, cmap='viridis')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.tight_layout()
    plt.savefig('/Users/soroush/Desktop/refs/figs/data_density.pdf', bbox_inches='tight')
    plt.close()

    # truer plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.contourf(X, Y, pdf_true_vals, levels=50, cmap='viridis', alpha=0.6)
    ax.quiver(X, Y, U_true, V_true, color='black', scale=scale_factor, scale_units='xy', 
             angles='xy', headwidth=5, width=0.003)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.tight_layout()
    plt.savefig('/Users/soroush/Desktop/refs/figs/true_score_field.pdf', bbox_inches='tight')
    plt.close()

    # learned score plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.contourf(X, Y, pdf_true_vals, levels=50, cmap='viridis', alpha=0.6)
    ax.quiver(X, Y, U_learned, V_learned, color='black', scale=scale_factor, scale_units='xy', 
             angles='xy', headwidth=5, width=0.003)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.tight_layout()
    plt.savefig('/Users/soroush/Desktop/refs/figs/learned_score_field.pdf', bbox_inches='tight')
    plt.close()

# LD -> comeback to this later
def langevin_dynamics_2d(score_model, num_samples=500, num_steps=200, epsilon=0.02, initial_points=None, grid_range=(-5, 5)):
    score_model.eval()

    if initial_points is None:
        x_t = (torch.rand(num_samples, 2) * (grid_range[1] - grid_range[0]) + grid_range[0])
    else:
        x_t = torch.tensor(initial_points, dtype=torch.float32)
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0)
        if x_t.shape[0] != num_samples:
            x_t = x_t.repeat(num_samples, 1)

    samples_history = [x_t.clone().detach().numpy()]

    for _ in range(num_steps):
        z_t = torch.randn_like(x_t)
        with torch.no_grad():
            score_val = score_model(x_t)
            score_val = torch.nan_to_num(score_val, nan=0.0, posinf=1.0, neginf=-1.0)
            x_t = x_t + (epsilon / 2) * score_val + torch.sqrt(torch.tensor(epsilon)) * z_t
        samples_history.append(x_t.clone().detach().numpy())

    return np.array(samples_history)

def visualize_langevin_oval(samples_history, title_suffix="", grid_range=(-5, 5)):
    final_samples = samples_history[-1, :, :]

    # grid for density plot
    x_density = np.linspace(grid_range[0], grid_range[1], 100)
    y_density = np.linspace(grid_range[0], grid_range[1], 100)
    X_density, Y_density = np.meshgrid(x_density, y_density)
    XY_density_tensor = torch.tensor(np.stack([X_density.ravel(), Y_density.ravel()], axis=-1), dtype=torch.float32)
    pdf_vals = pdf_oval(XY_density_tensor).detach().numpy().reshape(X_density.shape)
    filename_suffix = title_suffix.lower().replace(" ", "_").replace("-", "_")

    # final sample
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.contourf(X_density, Y_density, pdf_vals, levels=50, cmap='viridis', alpha=0.7)
    ax.scatter(final_samples[:, 0], final_samples[:, 1], s=5, alpha=1, color='black')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(grid_range)
    ax.set_ylim(grid_range)
    plt.tight_layout()
    plt.savefig(f'/Users/soroush/Desktop/refs/figs/langevin_final_samples_{filename_suffix}.pdf', bbox_inches='tight')
    plt.close()

    # trajectories
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.contour(X_density, Y_density, pdf_vals, levels=10, cmap='Greys', alpha=0.8)
    
    num_chains = min(10, samples_history.shape[1])
    for i in range(num_chains):
        ax.plot(samples_history[:, i, 0], samples_history[:, i, 1], alpha=0.6)
        # mark start and end points
        ax.scatter(samples_history[0, i, 0], samples_history[0, i, 1], marker='o', s=30, 
                  edgecolor='black', facecolor=plt.gca().lines[-1].get_color(), zorder=10)
        ax.scatter(samples_history[-1, i, 0], samples_history[-1, i, 1], marker='x', s=50, 
                  color='black', zorder=10)
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(grid_range)
    ax.set_ylim(grid_range)
    plt.tight_layout()
    plt.savefig(f'/Users/soroush/Desktop/refs/figs/langevin_trajectories_{filename_suffix}.pdf', bbox_inches='tight')
    plt.close()

def visualize_scores_oval2(model, grid_range=(-4, 4), n_points=40, density_band=(10, 70)):
    x = np.linspace(grid_range[0], grid_range[1], n_points)
    y = np.linspace(grid_range[0], grid_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=-1)
    XY_tensor = torch.tensor(XY, dtype=torch.float32)

    # true pdf for heatmap
    pdf_true_vals = pdf_oval(XY_tensor).detach().numpy().reshape(X.shape)

    # density thresholds for the band
    pdf_flat = pdf_true_vals.flatten()
    pdf_nonzero = pdf_flat[pdf_flat > 1e-10]  # avoiding numerical zeros
    lower_threshold = np.percentile(pdf_nonzero, density_band[0])
    upper_threshold = np.percentile(pdf_nonzero, density_band[1])

    # make the mask for medium-density regions (approaching the ring)
    medium_density_mask = (pdf_true_vals > lower_threshold) & (pdf_true_vals < upper_threshold)

    # true Scores
    scores_true_vals = true_score_oval(XY_tensor).detach().numpy()
    U_true, V_true = scores_true_vals[:, 0].reshape(X.shape), scores_true_vals[:, 1].reshape(X.shape)

    # learned Scores
    model.eval()
    with torch.no_grad():
        scores_learned_vals = model(XY_tensor).numpy()
    U_learned, V_learned = scores_learned_vals[:, 0].reshape(X.shape), scores_learned_vals[:, 1].reshape(X.shape)

    # we again need a common scale based on true scores
    # only consider vectors in the medium density band for scale calculation otherwiae boom
    true_magnitudes = np.sqrt(U_true[medium_density_mask]**2 + V_true[medium_density_mask]**2)
    common_scale = np.median(true_magnitudes) * 10

    # data density with band region
    fig, ax = plt.subplots(figsize=(7, 7))
    contourf = ax.contourf(X, Y, pdf_true_vals, levels=50, cmap='viridis')
    # show the density band with contour lines
    ax.contour(X, Y, pdf_true_vals, levels=[lower_threshold], colors='red', linewidths=2, linestyles='--', alpha=0.7)
    ax.contour(X, Y, pdf_true_vals, levels=[upper_threshold], colors='white', linewidths=2, linestyles='--', alpha=0.7)
    # show the band region
    band_highlight = np.where(medium_density_mask, 1, 0)
    ax.contourf(X, Y, band_highlight, levels=[0.5, 1.5], colors=['none', 'yellow'], alpha=0.2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.tight_layout()
    plt.savefig('/Users/soroush/Desktop/refs/figs/data_density_with_band.pdf', bbox_inches='tight')
    plt.close()
    # FINALLY this is FIXED!

    #true score medium band
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.contourf(X, Y, pdf_true_vals, levels=50, cmap='viridis', alpha=0.6)
    
    # vectors only in medium density band
    skip = 1
    X_band = np.where(medium_density_mask, X, np.nan)
    Y_band = np.where(medium_density_mask, Y, np.nan)
    U_true_band = np.where(medium_density_mask, U_true, np.nan)
    V_true_band = np.where(medium_density_mask, V_true, np.nan)
    
    ax.quiver(X_band[::skip, ::skip], Y_band[::skip, ::skip], 
             U_true_band[::skip, ::skip], V_true_band[::skip, ::skip],
             color='black', scale=common_scale, scale_units='xy', angles='xy', 
             headwidth=5, width=0.003, alpha=0.8)
    
    # mark the band boundaries
    ax.contour(X, Y, pdf_true_vals, levels=[lower_threshold, upper_threshold], 
              colors=['red', 'white'], linewidths=1.5, linestyles='--', alpha=0.7)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.tight_layout()
    plt.savefig('/Users/soroush/Desktop/refs/figs/true_score_medium_density.pdf', bbox_inches='tight')
    plt.close()

    # learned Score in medium abnd
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.contourf(X, Y, pdf_true_vals, levels=50, cmap='viridis', alpha=0.6)
    
    # learned vectors in same band
    U_learned_band = np.where(medium_density_mask, U_learned, np.nan)
    V_learned_band = np.where(medium_density_mask, V_learned, np.nan)
    
    ax.quiver(X_band[::skip, ::skip], Y_band[::skip, ::skip], 
             U_learned_band[::skip, ::skip], V_learned_band[::skip, ::skip],
             color='black', scale=common_scale, scale_units='xy', angles='xy', 
             headwidth=5, width=0.003, alpha=0.8)
    
    # mark the band boundaries
    ax.contour(X, Y, pdf_true_vals, levels=[lower_threshold, upper_threshold], 
              colors=['red', 'white'], linewidths=1.5, linestyles='--', alpha=0.7)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.tight_layout()
    plt.savefig('/Users/soroush/Desktop/refs/figs/learned_score_medium_density.pdf', bbox_inches='tight')
    plt.close()

    # stats for the medium density band
    band_indices = np.where(medium_density_mask)
    if len(band_indices[0]) > 0:
        # angular error - seems sus
        dot_product = (U_true[band_indices] * U_learned[band_indices] + 
                      V_true[band_indices] * V_learned[band_indices])
        mag_true = np.sqrt(U_true[band_indices]**2 + V_true[band_indices]**2)
        mag_learned = np.sqrt(U_learned[band_indices]**2 + V_learned[band_indices]**2)
        cos_angle = np.clip(dot_product / (mag_true * mag_learned + 1e-8), -1, 1)
        angle_errors = np.degrees(np.arccos(cos_angle))

        print(f"\nStatistics for medium-density band:")
        print(f" Number of vectors: {len(band_indices[0])}")
        print(f" Mean angle error: {angle_errors.mean():.1f}")
        print(f" Max angle error: {angle_errors.max():.1f}")
        print(f" Vectors with error > 45: {np.sum(angle_errors > 45)} ({100*np.mean(angle_errors > 45):.1f}%)")

    # some print stuff for double verification
    print(f"True score magnitude (mean): {np.nanmean(np.sqrt(U_true_band**2 + V_true_band**2)):.3f}")
    print(f"Learned score magnitude (mean): {np.nanmean(np.sqrt(U_learned_band**2 + V_learned_band**2)):.3f}")

    # check if magnitudes match
    true_mag = np.mean(np.sqrt(U_true[medium_density_mask]**2 + V_true[medium_density_mask]**2))
    learned_mag = np.mean(np.sqrt(U_learned[medium_density_mask]**2 + V_learned[medium_density_mask]**2))
    print(f"Magnitude ratio: {learned_mag/true_mag:.2f}x")


def plot_training_loss(epoch_losses):
    fig, ax = plt.subplots(figsize=(7, 7))
    epochs = range(1, len(epoch_losses) + 1)
    ax.plot(epochs, epoch_losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Loss per Epoch')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/soroush/Desktop/refs/figs/training_loss_ex4.pdf', bbox_inches='tight')
    plt.close()


# main
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    # train the model
    trained_model, losses = train_with_score_matching_oval()

    error_analysis = analyze_score_errors(
        trained_model,
        true_score_oval,
        pdf_oval
    )

    plot_training_loss(losses)

    visualize_scores_oval(trained_model)

    low_density_start = np.array([[0.0, 0.0]])
    langevin_samples_low = langevin_dynamics_2d(trained_model, initial_points=low_density_start)
    visualize_langevin_oval(langevin_samples_low, title_suffix="started in low-density center")

    high_density_start = sampler_oval(500).numpy()  # Start with multiple samples
    langevin_samples_high = langevin_dynamics_2d(trained_model, num_samples=500, initial_points=high_density_start)
    visualize_langevin_oval(langevin_samples_high, title_suffix="started in high-density ting")

    outer_low_density_start = np.array([[4.0, 4.0]])
    langevin_samples_outer = langevin_dynamics_2d(trained_model, initial_points=outer_low_density_start)
    visualize_langevin_oval(langevin_samples_outer, title_suffix="started in outer low-density region")

    visualize_scores_oval2(trained_model, density_band=(45, 80))
    #visualize_scores_oval2(trained_model, density_band=(90, 100))