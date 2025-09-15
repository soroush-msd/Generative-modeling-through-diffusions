import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = "/Users/soroush/Desktop/refs/figs"  # change if needed
os.makedirs(output_dir, exist_ok=True)

# define mixture parameters for 2D Gaussian mixture
weights = np.array([1/3, 1/3, 1/3])
means = [
    np.array([-3.0, 0.0]),
    np.array([3.0, 0.0]),
    np.array([0.0, 2.0])
]
covs = [
    0.5 * np.eye(2),
    0.5 * np.eye(2),
    0.5 * np.eye(2)
]

def gaussian_pdf(x, mean, cov):
    x = np.atleast_2d(x)
    dim = x.shape[1]
    cov_inv = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    norm_const = 1.0 / (np.sqrt((2*np.pi)**dim * det_cov))
    diff = x - mean
    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
    return norm_const * np.exp(exponent)

def mixture_pdf(x, weights, means, covs):
    x = np.atleast_2d(x)
    pdf_vals = np.zeros(x.shape[0])
    for w, m, c in zip(weights, means, covs):
        pdf_vals += w * gaussian_pdf(x, m, c)
    return pdf_vals

def mixture_logpdf(x, weights, means, covs):
    return np.log(mixture_pdf(x, weights, means, covs) + 1e-15)

def mixture_score(x, weights, means, covs):
    x = np.atleast_2d(x)
    pdf_val = mixture_pdf(x, weights, means, covs).reshape(-1, 1)
    grads = np.zeros_like(x)
    for w, m, c in zip(weights, means, covs):
        c_inv = np.linalg.inv(c)
        diff = (x - m)
        comp_pdf = gaussian_pdf(x, m, c).reshape(-1, 1)
        grads_comp = (-comp_pdf * (diff @ c_inv.T))
        grads += w * grads_comp
    grads = grads / (pdf_val + 1e-15)
    return grads

# set up grid for visualization
grid_size = 30
x_lin_vec = np.linspace(-6, 6, grid_size)
y_lin_vec = np.linspace(-6, 6, grid_size)
xx_vec, yy_vec = np.meshgrid(x_lin_vec, y_lin_vec)
xy_grid_vec = np.column_stack([xx_vec.ravel(), yy_vec.ravel()])
grad_field = mixture_score(xy_grid_vec, weights, means, covs)
u = grad_field[:, 0].reshape(xx_vec.shape)
v = grad_field[:, 1].reshape(xx_vec.shape)
magnitude = np.sqrt(u**2 + v**2)
max_mag = np.max(magnitude)
u_norm = u / (max_mag + 1e-10)
v_norm = v / (max_mag + 1e-10)

grid_size_contour = 100
x_lin = np.linspace(-6, 6, grid_size_contour)
y_lin = np.linspace(-6, 6, grid_size_contour)
xx, yy = np.meshgrid(x_lin, y_lin)
xy_grid = np.column_stack([xx.ravel(), yy.ravel()])
z = mixture_pdf(xy_grid, weights, means, covs).reshape(xx.shape)

# langevin sampling parameters 
num_particles = 500
num_steps = 24
epsilon = 0.05

np.random.seed(42)
particles = np.random.uniform(-5.5, 5.5, size=(num_particles, 2))
trajectories = [particles.copy()]

# langevin dynamics
for step in range(num_steps):
    grad_logp = mixture_score(particles, weights, means, covs)
    noise = np.random.randn(num_particles, 2)
    particles = particles + epsilon * grad_logp + np.sqrt(2 * epsilon) * noise
    trajectories.append(particles.copy())

# save 3 key frames
step_indices = [0, num_steps // 4, num_steps]
step_names = ["start", "middle", "end"]

for idx, name in zip(step_indices, step_names):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    # contour plot of the mixture density
    contour = ax.contourf(xx, yy, z, levels=30, cmap='viridis')
    # vector field (score)
    arrow_stride = 2
    ax.quiver(xx_vec[::arrow_stride, ::arrow_stride],
              yy_vec[::arrow_stride, ::arrow_stride],
              u_norm[::arrow_stride, ::arrow_stride],
              v_norm[::arrow_stride, ::arrow_stride],
              magnitude[::arrow_stride, ::arrow_stride],    # use magnitude for color
              cmap='turbo', scale=25, width=0.003, alpha=0.7)
              
    # particles at this step
    ax.scatter(trajectories[idx][:, 0], trajectories[idx][:, 1], c='white', edgecolors='black', s=20)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    #ax.set_title(f"Langevin Dynamics ({name.capitalize()})")
    fig.tight_layout()
    # save as vectorized pdf
    fig.savefig(f"{output_dir}/langevin_mixture_{name}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

# animate process for interactive exploration
import matplotlib.animation as animation
plt.rcParams['animation.html'] = 'jshtml'

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
contour = ax.contourf(xx, yy, z, levels=30, cmap='viridis')
arrow_stride = 2
quiver = ax.quiver(xx_vec[::arrow_stride, ::arrow_stride],
                   yy_vec[::arrow_stride, ::arrow_stride],
                   u_norm[::arrow_stride, ::arrow_stride],
                   v_norm[::arrow_stride, ::arrow_stride],
                   color='black', scale=25, width=0.003, alpha=0.7)
particles_scatter = ax.scatter([], [], c='white', edgecolors='black', s=20)
frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top', fontsize=10)
def init():
    particles_scatter.set_offsets(np.empty((0, 2)))
    frame_text.set_text('')
    return particles_scatter, quiver, frame_text
def update(frame):
    particles_scatter.set_offsets(trajectories[frame])
    frame_text.set_text(f'Step: {frame}/{num_steps}')
    return particles_scatter, quiver, frame_text
ani = animation.FuncAnimation(fig, update, frames=len(trajectories),
                             init_func=init, blit=False, interval=200)
plt.title("langevin Dynamics on 3, 2D-Gaussian Mixture")
plt.tight_layout()
plt.show()
_ani_ref = ani  # prevent garbage collection

