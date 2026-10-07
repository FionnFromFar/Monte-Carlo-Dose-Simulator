import numpy as np
import matplotlib.pyplot as plt

def run_monte_carlo_pdd_v2(num_particles=20000, mu=0.15, phantom_depth=20.0, num_bins=40, initial_energy=1.0):
    """
    Version 0.2: Explicit particle tracking with multi-step interactions
    and fractional energy deposition.
    """
    # Spatial grid setup
    bin_width = phantom_depth / num_bins
    bin_edges = np.linspace(0, phantom_depth, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    # Initialize scorekeeper for dose deposition across bins
    dose_profile = np.zeros(num_bins)
    
    # Simulate individual particle histories
    for i in range(num_particles):
        z = 0.0  # Photon starts at the surface of the water phantom
        E = initial_energy  # Initial photon energy
        
        # Transport loop: track photon while it is inside the phantom and has energy
        while 0.0 <= z < phantom_depth and E > 0.05:
            # 1. Sample interaction distance using exponential attenuation (mean free path)
            xi = np.random.uniform(0.0001, 1.0)
            s = -np.log(xi) / mu
            
            # 2. Advance photon position to the interaction site
            z += s
            
            # 3. Check if interaction happens inside the phantom boundary
            if 0.0 <= z < phantom_depth:
                # Fractional energy deposition (simulating energy transferred to recoil electrons)
                dep_fraction = 0.3
                deposited_energy = E * dep_fraction
                
                # Map interaction depth to the correct spatial bin index
                bin_idx = int(z / bin_width)
                if 0 <= bin_idx < num_bins:
                    dose_profile[bin_idx] += deposited_energy
                    
                # Reduce remaining photon energy due to scattering/absorption
                E -= deposited_energy
            else:
                # Photon exited the back of the phantom tank
                break
                
    # Normalize the dose profile so the maximum value is 100%
    max_dose = np.max(dose_profile)
    if max_dose > 0:
        pdd_curve = (dose_profile / max_dose) * 100.0
    else:
        pdd_curve = dose_profile
        
    return bin_centers, pdd_curve

# --- Execute Simulation ---
bin_centers, pdd = run_monte_carlo_pdd_v2()

# --- Plotting the Results ---
plt.figure(figsize=(10, 5))
plt.plot(bin_centers, pdd, marker='o', linestyle='-', color='crimson', linewidth=2, label='Monte Carlo v0.2 Model')
plt.title('Percent Depth Dose (PDD) - Multi-Step Tracking Model', fontsize=14, fontweight='bold')
plt.xlabel('Depth in Water Phantom (cm)', fontsize=12)
plt.ylabel('Relative Dose (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()