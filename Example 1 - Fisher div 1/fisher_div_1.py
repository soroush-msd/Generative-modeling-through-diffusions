import numpy as np
import matplotlib.pyplot as plt
import os

# output folder - change this
output_dir = "/Users/soroush/Desktop/refs/figs"
os.makedirs(output_dir, exist_ok=True)

np.random.seed(42)
# true data distribution parameter
sigma_d2 = 1.0

# model distribution variance parameter values to test
sigma2_values = np.linspace(0.1, 3.0, 500)

# main functions
# the true loss
def fisher_divergence_loss(sigma_d2, sigma2):
    return 0.5 * sigma_d2 * ((1 / sigma_d2) - (1 / sigma2))**2

# the score matching objective
def j_theta(sigma_d2, sigma2):
    return ((sigma_d2**2) / (2 * sigma2**4)) - (1 / sigma2**2)

# the estimator for a single sample.
def j_hat_theta(samples, sigma2):
    gradient_log_p = -samples / sigma2
    trace_hessian_log_p = -1 / sigma2
    
    term1 = trace_hessian_log_p
    term2 = 0.5 * np.mean(gradient_log_p**2, axis=0)
    return (term1 + term2)

# simulation parameters
sample_sizes = [10, 50, 100, 500]
num_simulations = 50 # nmber of independent experiments to run for each n, for variability

for N in sample_sizes:
    # store the results of all j_hat curves from the simulations
    all_j_hat_curves = []
    
    # monte carlo simulation loop
    for i in range(num_simulations):
        # draw a fresh ind sample from the true distribution
        samples = np.random.normal(0, np.sqrt(sigma_d2), size=N)
        
        # calculate the j_hat curve for this one sample
        current_j_hat_curve = j_hat_theta(samples.reshape(-1, 1), sigma2_values)
        all_j_hat_curves.append(current_j_hat_curve)
        
    # convert list of curves to a numpy array for easy stats
    all_j_hat_curves = np.array(all_j_hat_curves)
    
    # calculate statistics across simulations
    mean_j_hat_curve = np.mean(all_j_hat_curves, axis=0)
    std_j_hat_curve = np.std(all_j_hat_curves, axis=0, ddof=1)

    # calculate analytical L and J these dont change
    L_vals = fisher_divergence_loss(sigma_d2, sigma2_values)
    J_vals = j_theta(sigma_d2, sigma2_values)
    
    # plotting the results
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.plot(sigma2_values, L_vals, label=f"$L(\\theta)$", linestyle='-', color='blue', linewidth=2)
    ax.plot(sigma2_values, J_vals, label=f"$J(\\theta)$", linestyle='-', color='green', linewidth=2)
    
    # plot the mean of the estimator
    ax.plot(sigma2_values, mean_j_hat_curve, label=f"Mean $\\hat{{J}}(\\theta)$", linestyle='-', color='red', linewidth=2)
    
    # add the shaded region for variability
    ax.fill_between(sigma2_values, 
                    mean_j_hat_curve - std_j_hat_curve, 
                    mean_j_hat_curve + std_j_hat_curve, 
                    color='red', alpha=0.2, label='$\\pm 1$ std. dev.')

    # find and plot the minimum points
    min_L_idx = np.argmin(L_vals)
    min_J_idx = np.argmin(J_vals)
    min_J_hat_idx = np.argmin(mean_j_hat_curve)

    ax.scatter(sigma2_values[min_L_idx], L_vals[min_L_idx],
               color='blue', s=100, zorder=5, ec='black', marker='o')
    ax.scatter(sigma2_values[min_J_idx], J_vals[min_J_idx],
               color='green', s=100, zorder=5, ec='black', marker='o')
    ax.scatter(sigma2_values[min_J_hat_idx], mean_j_hat_curve[min_J_hat_idx],
               color='red', s=100, zorder=5, ec='black', marker='o')

    #ax.set_title(f"Sample Size N={N}")
    ax.set_xlabel("Model Variance $\\theta$")
    ax.set_ylabel("Loss Value")
    ax.legend()
    ax.grid(True, linestyle=':')
    ax.set_ylim(-1.5, 4) # adjust y-axis for better visibility
    
    fig.tight_layout()
    
    # save each plot as its own file
    fig.savefig(f"{output_dir}/j_hat_sampling_variability_N{N}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)  # close to free up memory
