import os
import numpy as np
import matplotlib.pyplot as plt

def run_monte_carlo_pdd_v4(num_particles=25000, phantom_depth=20.0, num_bins=40, initial_energy=1.0):
    """
    Version 0.4: Simulates secondary electron transport by applying a forward 
    displacement shift to model the surface buildup region (Kerma vs. Dose).
    """
    bin_width = phantom_depth / num_bins
    bin_edges = np.linspace(0, phantom_depth, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    dose_profile = np.zeros(num_bins)
    
    for i in range(num_particles):
        z = 0.0
        E = initial_energy
        
        # Track photon while inside the phantom and above energy cutoff
        while 0.0 <= z < phantom_depth and E > 0.02:
            # Energy-dependent attenuation coefficient
            mu = 0.1 + 0.04 / (E + 0.1)
            
            # Sample photon interaction distance
            xi = np.random.uniform(0.0001, 1.0)
            s = -np.log(xi) / mu
            
            z += s
            
            if 0.0 <= z < phantom_depth:
                # Determine interaction type probabilities (Photoelectric vs. Compton)
                pe_prob = 0.06 / (E + 0.05)
                pe_prob = np.clip(pe_prob, 0.05, 0.85)
                
                interaction_roll = np.random.uniform(0.0, 1.0)
                
                if interaction_roll < pe_prob or E < 0.1:
                    # Photoelectric Effect: Full energy deposition
                    deposited_energy = E
                    E = 0.0  # Terminate photon history
                else:
                    # Compton Scattering: Partial energy transfer to recoil electron
                    transfer_fraction = np.random.uniform(0.1, 0.5)
                    deposited_energy = E * transfer_fraction
                    E -= deposited_energy
                
                # Secondary Electron Transport Shift
                delta_x = np.random.exponential(scale=0.6)  # Mean projected forward electron range
                z_dose = z + delta_x
                
                # Map the shifted dose deposition depth to spatial bins
                bin_idx = int(z_dose / bin_width)
                if 0 <= bin_idx < num_bins:
                    dose_profile[bin_idx] += deposited_energy
                    
                if E == 0.0:
                    break
            else:
                # Photon exited the phantom
                break
                
    # Normalize the dose profile so the maximum value is 100%
    max_dose = np.max(dose_profile)
    if max_dose > 0:
        pdd_curve = (dose_profile / max_dose) * 100.0
    else:
        pdd_curve = dose_profile
        
    return bin_centers, pdd_curve

# --- Execute Simulation ---
bin_centers, pdd = run_monte_carlo_pdd_v4()

# --- Ensure the Figures directory exists ---
os.makedirs('Figures', exist_ok=True)

# --- Plotting and Saving the Results ---
plt.figure(figsize=(10, 5))
plt.plot(bin_centers, pdd, marker='o', linestyle='-', color='dodgerblue', linewidth=2, label='Monte Carlo v0.4 Model')
plt.title('Percent Depth Dose (PDD) - Buildup Region Model (v0.4)', fontsize=14, fontweight='bold')
plt.xlabel('Depth in Water Phantom (cm)', fontsize=12)
plt.ylabel('Relative Dose (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# Save the figure to your Figures folder (must be called BEFORE plt.show())
plt.savefig('Figures/v0.4 PDD curve.png', dpi=300)

# Display the interactive plot window
plt.show()