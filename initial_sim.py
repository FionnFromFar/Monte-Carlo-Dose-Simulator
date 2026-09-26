import numpy as np

def sample_interaction_distance(mu, num_particles=1):
    """
    Samples the distance a particle travels before its next interaction 
    using Inverse Transform Sampling of the exponential decay law.
    """
    # Generate flat uniform random numbers between 0 and 1
    # Using 0.0001 instead of 0.0 because ln(0) is negative infinity 
    xi = np.random.uniform(0.0001, 1.0, size=num_particles)
    
    # Apply the inverse transform formula: s = -ln(xi) / mu
    distance = -np.log(xi) / mu
    
    return distance

def setup_phantom_and_bin_distances(distances, phantom_depth=20.0, num_bins=40):
    """
    Takes sampled interaction distances and sorts them into spatial bins 
    representing a 1D water phantom.
    """
    # Create spatial grid boundaries 
    bin_edges = np.linspace(0, phantom_depth, num_bins + 1)
    
    # Filter out particles that travel past the end of our 20 cm tank
    # (In real medical physics, these particles exit the patient/phantom)
    valid_particles = distances < phantom_depth
    valid_distances = distances[valid_particles]
    
    # Using np.digitize to find which bin index each distance falls into
    # np.digitize compares each distance against our bin_edges and returns the bin number
    bin_indices = np.digitize(valid_distances, bin_edges) - 1
    
    # Step 4: Initialize an array of zeros to act as our scorekeeper (dose profile)
    dose_profile = np.zeros(num_bins)
    
    # Step 5: Tally up the interactions in each bin
    for idx in bin_indices:
        if 0 <= idx < num_bins:  # Safety check to ensure it's inside our grid
            dose_profile[idx] += 1
            
    return bin_edges, dose_profile


# Testing

# Simulation parameters
num_particles = 50000
mu_water = 0.15       # Attenuation coefficient for water (1/cm)
phantom_depth = 20.0  # Total depth of our water tank (cm)
num_bins = 40         # Number of spatial slices

# Sampling interaction distances for all particles
distances = sample_interaction_distance(mu_water, num_particles=num_particles)

# Feeding them into the functions
bin_edges, dose_profile = setup_phantom_and_bin_distances(distances, phantom_depth, num_bins)

# Getting the results back
print("First 5 spatial bin edges (cm):", np.round(bin_edges[:6], 2))
print("Interactions scored in the first 5 bins:", dose_profile[:5].astype(int)) # for the first 5 cos 50k is too much
print("Total particles that interacted inside the 20cm phantom:", int(np.sum(dose_profile)))