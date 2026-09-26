import numpy as np
import matplotlib.pyplot as plt

def run_monte_carlo_pdd(num_particles=50000, mu=0.15, phantom_depth=20.0, num_bins=40, initial_energy=1.0):
    """
    Simulates 1D photon transport and scores energy deposition to create a depth-dose profile.
    """
    # Spatial grid
    bin_edges = np.linspace(0, phantom_depth, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # Centers of each bin for plotting
    
    # Initialize scorekeeper
    dose_profile = np.zeros(num_bins)
    
    # Sample interaction distances for all particles at once
    xi = np.random.uniform(0.0001, 1.0, size=num_particles)
    distances = -np.log(xi) / mu
    
    # Filter out particles that exit the back of the 20 cm phantom tank
    valid_particles = distances < phantom_depth
    valid_distances = distances[valid_particles]
    
    # Map valid interaction distances into spatial bin indices
    bin_indices = np.digitize(valid_distances, bin_edges) - 1
    
    # 5. Score energy deposition: Assume every particle deposits its full initial energy upon interaction
    for idx in bin_indices:
        if 0 <= idx < num_bins:
            dose_profile[idx] += initial_energy
            
    # Normalize the dose profile so the maximum value is 100% (Standard PDD representation)
    max_dose = np.max(dose_profile)
    if max_dose > 0:
        pdd_curve = (dose_profile / max_dose) * 100.0
    else:
        pdd_curve = dose_profile
        
    return bin_centers, pdd_curve

# --- Execute Simulation ---
bin_centers, pdd = run_monte_carlo_pdd()

# --- Plotting the Results ---
plt.figure(figsize=(10, 5))
plt.plot(bin_centers, pdd, marker='o', linestyle='-', color='teal', linewidth=2, label='Monte Carlo 1D Model')
plt.title('Simulated Percent Depth Dose (PDD) Curve', fontsize=14, fontweight='bold')
plt.xlabel('Depth in Water Phantom (cm)', fontsize=12)
plt.ylabel('Relative Dose (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# Display the interactive plot window
plt.show()