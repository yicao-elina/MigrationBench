import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Load the NEB data
neb_data_file = "sb2te3.dat"
data = np.loadtxt(neb_data_file)

# Extract reaction coordinate and energy
reaction_coordinate = data[:, 0]
energy = data[:, 1]

# Interpolate for smooth curve
x_smooth = np.linspace(reaction_coordinate.min(), reaction_coordinate.max(), 500)
spline = make_interp_spline(reaction_coordinate, energy, k=3)  # cubic spline
energy_smooth = spline(x_smooth)

# Plot
plt.figure(figsize=(6, 4))
plt.plot(x_smooth, energy_smooth, label='NEB Energy Path', color='b')
plt.plot(reaction_coordinate, energy, 'o', color='black')  # show raw points
plt.xlabel("Reaction Coordinate")
plt.ylabel("Energy (eV)")
plt.title("NEB Energy Profile (Smoothed)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("neb_energy_barrier.pdf")
plt.show()
