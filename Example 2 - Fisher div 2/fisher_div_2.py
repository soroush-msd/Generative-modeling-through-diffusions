import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

output_dir = "/Users/soroush/Desktop/refs/figs"   # path, change this
os.makedirs(output_dir, exist_ok=True)

np.random.seed(42)
# true parameters
mu_d = 5
sigma_d2 = 1

def L(mu, sigma2):
    return ((sigma2 - sigma_d2)**2) / (2 * sigma_d2 * sigma2**2) + (mu - mu_d)**2 / (2 * sigma2**2)

def J(mu, sigma2):
    return ((mu - mu_d)**2 + sigma_d2 - 2 * sigma2) / (2 * sigma2**2)

def J_hat(mu, sigma2, samples):
    x_bar = np.mean(samples)
    x_sq_bar = np.mean(samples**2)
    return -1/sigma2 + (mu**2 - 2*mu*x_bar + x_sq_bar) / (2 * sigma2**2)

mu = np.linspace(3, 7, 100)
sigma2 = np.linspace(0.25, 2, 100)
Mu, Sigma2 = np.meshgrid(mu, sigma2)

sample_sizes = [10, 50, 100, 500]

for N in sample_sizes:
    samples = np.random.normal(mu_d, np.sqrt(sigma_d2), N)

    # plots for L
    #L_vals = L(Mu, Sigma2)
    #min_L_idx = np.unravel_index(np.argmin(L_vals), L_vals.shape)
    #min_L = (Mu[min_L_idx], Sigma2[min_L_idx])
    #fig = plt.figure(figsize=(6, 6))
    #ax = fig.add_subplot(111, projection='3d')
    #surf = ax.plot_surface(Mu, Sigma2, L_vals, cmap='viridis', alpha=0.8)
    #ax.contour(Mu, Sigma2, L_vals, zdir='z', offset=np.min(L_vals), cmap='viridis', linestyles='dashed')
    #ax.scatter(mu_d, sigma_d2, L(mu_d, sigma_d2), color='red', s=100, marker='x', label='True Parameters')
    #ax.scatter(*min_L, L(*min_L), color='black', s=50, marker='o', label='Minimum')
    #ax.view_init(elev=30, azim=-60)
    #ax.set_xlabel(r'$\mu$')
    #ax.set_ylabel(r'$\sigma^2$')
    #ax.set_zlabel(r'$L(\theta)$')
    ##ax.legend()
    #plt.tight_layout()
    #fig.savefig(f"{output_dir}/loss_surface_L_N{N}.pdf", bbox_inches='tight', format='pdf')
    #plt.close(fig)

    # plots for J
    #J_vals = J(Mu, Sigma2)
    #min_J_idx = np.unravel_index(np.argmin(J_vals), J_vals.shape)
    #min_J = (Mu[min_J_idx], Sigma2[min_J_idx])
    #fig = plt.figure(figsize=(6, 6))
    #ax = fig.add_subplot(111, projection='3d')
    #surf = ax.plot_surface(Mu, Sigma2, J_vals, cmap='plasma', alpha=0.8)
    #ax.contour(Mu, Sigma2, J_vals, zdir='z', offset=np.min(J_vals), cmap='plasma', linestyles='dashed')
    #ax.scatter(mu_d, sigma_d2, J(mu_d, sigma_d2), color='red', s=100, marker='x', label='True Parameters')
    #ax.scatter(*min_J, J(*min_J), color='black', s=50, marker='o', label='Minimum')
    #ax.view_init(elev=30, azim=-60)
    #ax.set_xlabel(r'$\mu$')
    #ax.set_ylabel(r'$\sigma^2$')
    #ax.set_zlabel(r'$J(\theta)$')
    #ax.legend()
    #plt.tight_layout()
    #fig.savefig(f"{output_dir}/loss_surface_J_N{N}.pdf", bbox_inches='tight', format='pdf')
    #plt.close(fig)

    # plots for the estimator
    J_hat_vals = J_hat(Mu, Sigma2, samples)
    min_J_hat_idx = np.unravel_index(np.argmin(J_hat_vals), J_hat_vals.shape)
    min_J_hat = (Mu[min_J_hat_idx], Sigma2[min_J_hat_idx])
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Mu, Sigma2, J_hat_vals, cmap='plasma', alpha=0.7)
    ax.contour(Mu, Sigma2, J_hat_vals, zdir='z', offset=np.min(J_hat_vals), cmap='plasma', linestyles='dashed')
    ax.scatter(mu_d, sigma_d2, J_hat(mu_d, sigma_d2, samples), color='red', s=100, marker='x', label='True Parameters')
    ax.scatter(*min_J_hat, J_hat(*min_J_hat, samples), color='black', s=50, marker='o', label='Minimum')
    ax.view_init(elev=30, azim=-60)
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$\sigma^2$')
    ax.set_zlabel(r'$\hat{J}(\theta)$')
    ax.legend(loc='upper left')
    plt.tight_layout()
    fig.savefig(f"{output_dir}/loss_surface_Jhat_N{N}.pdf", bbox_inches='tight', format='pdf')
    plt.close(fig)

