import os
import numpy as np
import matplotlib.pyplot as plt

def get_continuous_spectrum_energy(E_max):
    """
    Samples a continuous Bremsstrahlung energy distribution based on the provided E_max.
    """
    peak_energy = E_max / 3.0
    max_prob = peak_energy * (E_max - peak_energy)**2
    
    while True:
        E_guess = np.random.uniform(0.1, E_max)
        P_guess = (E_guess * (E_max - E_guess)**2) / max_prob
        
        if np.random.uniform(0.0, 1.0) < P_guess:
            return E_guess

def run_monte_carlo_pdd(num_particles=100000, phantom_depth=30.0, num_bins=60, E_max=6.0):
    """
    Runs the v0.6 transport kernel for a specified maximum beam energy (E_max).
    """
    bin_width = phantom_depth / num_bins
    bin_edges = np.linspace(0, phantom_depth, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    dose_profile = np.zeros(num_bins)
    
    for i in range(num_particles):
        z = 0.0
        E = get_continuous_spectrum_energy(E_max)
        
        while 0.0 <= z < phantom_depth and E > 0.02:
            mu = 0.12 + 0.05 / (E + 0.1)
            
            xi = np.random.uniform(0.0001, 1.0)
            s = -np.log(xi) / mu
            z += s
            
            if 0.0 <= z < phantom_depth:
                pe_prob = 0.06 / (E + 0.05)
                pe_prob = np.clip(pe_prob, 0.05, 0.85)
                
                interaction_roll = np.random.uniform(0.0, 1.0)
                
                if interaction_roll < pe_prob or E < 0.1:
                    deposited_energy = E
                    E_for_electron = E  
                    E = 0.0  
                else:
                    transfer_fraction = np.random.uniform(0.2, 0.65)
                    deposited_energy = E * transfer_fraction
                    E_for_electron = deposited_energy 
                    E -= deposited_energy
                    
                    escape_chance = 0.3 + 0.1 / (E + 0.1) 
                    if np.random.uniform() < escape_chance:
                        E = 0.0  
                
                # Energy-Coupled Electron Shift
                electron_range_scale = 0.3 * E_for_electron
                delta_x = np.random.exponential(scale=electron_range_scale)
                z_dose = z + delta_x
                
                bin_idx = int(z_dose / bin_width)
                if 0 <= bin_idx < num_bins:
                    dose_profile[bin_idx] += deposited_energy
                    
                if E == 0.0:
                    break
            else:
                break
                
    max_dose = np.max(dose_profile)
    if max_dose > 0:
        pdd_curve = (dose_profile / max_dose) * 100.0
    else:
        pdd_curve = dose_profile
        
    return bin_centers, pdd_curve

# Standard clinical linac energies to simulate
beam_energies = [4.0, 6.0, 10.0, 15.0]
colors = ['teal', 'darkorange', 'crimson', 'purple']

# Ensure the new nested directory exists
output_dir = os.path.join('Figures', 'final version dose distributions')
os.makedirs(output_dir, exist_ok=True)

# --- Plotting Setup ---
plt.figure(figsize=(12, 6))

# Run simulation for each energy and plot on the same graph
for E_max, color in zip(beam_energies, colors):
    print(f"Simulating {E_max} MV beam...")
    bin_centers, pdd = run_monte_carlo_pdd(E_max=E_max)
    plt.plot(bin_centers, pdd, marker='o', markersize=4, linestyle='-', color=color, linewidth=2, label=f'{int(E_max)} MV Beam')

# --- Formatting Final Figure ---
plt.title('PDD Comparison by Energy (v0.6)', fontsize=15, fontweight='bold')
plt.xlabel('Depth in Water Phantom (cm)', fontsize=12)
plt.ylabel('Relative Dose (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.xlim(0, 30)
plt.tight_layout()

# Save the combined figure
save_path = os.path.join(output_dir, 'Multi_Energy_PDD_Comparison.png')
plt.savefig(save_path, dpi=300)
print(f"\nFigure saved successfully to: {save_path}")

plt.show()