import numpy as np
import matplotlib.pyplot as plt

def run_monte_carlo_pdd_v3(num_particles=20000, phantom_depth=20.0, num_bins=40, initial_energy=1.0):
    """
    Version 0.3: Multi-interaction model simulating Compton scattering 
    and the Photoelectric effect with probabilistic branching.
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
            # 1. Energy-dependent attenuation coefficient (mu increases as energy drops)
            mu = 0.1 + 0.04 / (E + 0.1)
            
            # 2. Sample interaction distance using mean free path
            xi = np.random.uniform(0.0001, 1.0)
            s = -np.log(xi) / mu
            
            z += s
            
            if 0.0 <= z < phantom_depth:
                # 3. Determine interaction type probabilities based on current energy
                # Photoelectric effect probability increases drastically at low energies
                pe_prob = 0.06 / (E + 0.05)
                pe_prob = np.clip(pe_prob, 0.05, 0.85)  # Bound probability between 5% and 85%
                
                interaction_roll = np.random.uniform(0.0, 1.0)
                bin_idx = int(z / bin_width)
                
                if interaction_roll < pe_prob or E < 0.1:
                    # Photoelectric Effect: Photon is completely absorbed (100% energy deposition)
                    deposited_energy = E
                    if 0 <= bin_idx < num_bins:
                        dose_profile[bin_idx] += deposited_energy
                    E = 0.0  # End photon history
                    break
                else:
                    # Compton Scattering: Partial energy transfer to recoil electron
                    # Transfer a random fraction (e.g., 10% to 50% of remaining energy)
                    transfer_fraction = np.random.uniform(0.1, 0.5)
                    deposited_energy = E * transfer_fraction
                    
                    if 0 <= bin_idx < num_bins:
                        dose_profile[bin_idx] += deposited_energy
                        
                    # Photon survives with reduced energy and continues its journey
                    E -= deposited_energy
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
bin_centers, pdd = run_monte_carlo_pdd_v3()

# --- Plotting the Results ---
plt.figure(figsize=(10, 5))
plt.plot(bin_centers, pdd, marker='o', linestyle='-', color='purple', linewidth=2, label='Monte Carlo v0.3 Model')
plt.title('Percent Depth Dose (PDD) - Multi-Interaction Model (v0.3)', fontsize=14, fontweight='bold')
plt.xlabel('Depth in Water Phantom (cm)', fontsize=12)
plt.ylabel('Relative Dose (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()