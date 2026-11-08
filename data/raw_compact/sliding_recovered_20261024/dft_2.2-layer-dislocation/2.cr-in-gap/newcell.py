from ase.io import read, write
import numpy as np

# Step 1: Read original supercell structure
structure = read('after_relax.xyz')

# Step 2: Define displacement vector 
# l(Te19-Sb15) =  2.99185(0) Å
#   19    Te19 Te   4.31789   0.01787  32.25608 ( 0  0  0)+ x, y, z
#   39    Sb15 Sb   6.42936   1.23693  30.52209 ( 0  0  0)+ x, y, z

#   9     Te9 Te   2.17004   1.23377  24.88883 ( 0  0  0)+ x, y, z
#    7     Te7 Te   4.31584   2.47243  22.37426 ( 0  0  0)+ x, y, z


Te19 = np.array([2.17004, 1.23377, 24.88883])
Sb15 = np.array([4.31584, 2.47243, 24.88883])

displacement_vector = Sb15 - Te19  # Vector components in Å


# l(Sb11-Sb34) =  7.31901(0) Å
#   47    Sb11 Sb   4.23200   0.01033  18.01717 ( 0  0  0)+ x, y, z
#  106    Sb34 Sb  10.57045   3.66983  18.02076 ( 0  0  0)+ x, y, z
Sb11 = np.array([4.23200, 0.01033, 18.01717])
Sb34 = np.array([10.57045, 3.66983, 18.02076])
displacement_vector_Sb = Sb11 - Sb34
# Optional: print the displacement vector to verify
print("Displacement vector (Å):", displacement_vector)
print("Displacement vector Sb (Å):", displacement_vector_Sb)

# Step 3: Identify upper half atoms (above midpoint along c-axis)
positions = structure.get_positions()
# z_mid = (Te19[2] + Sb15[2]) / 2.0
z_mid = 23.7 # Cr z-coordinate

# Displace atoms above the midpoint
new_positions = positions.copy()
new_positions_Sb = positions.copy()
for i, pos in enumerate(positions):
    if pos[2] > z_mid:
        new_positions[i] += displacement_vector  # Move by the vector
        new_positions_Sb[i] += displacement_vector_Sb
        

# Apply periodic boundary conditions
# cell = structure.get_cell()
# new_positions = np.mod(new_positions, cell.diagonal())
# double_new_positions = np.mod(double_new_positions, cell.diagonal())

# Step 4: Create and save the new displaced structure
displaced_structure = structure.copy()
displaced_structure.set_positions(new_positions)

displaced_structure_Sb = structure.copy()
displaced_structure_Sb.set_positions(new_positions_Sb)

# Export new structure
write('Sb2Te3_dislocated_supercell.xyz', displaced_structure, format='extxyz')
write('Sb2Te3_dislocated_supercell.in', displaced_structure, format='espresso-in')

write('Sb2Te3_dislocated_supercell_Sb.xyz', displaced_structure_Sb, format='extxyz')
write('Sb2Te3_dislocated_supercell_Sb.in', displaced_structure_Sb, format='espresso-in')
print("Displaced structure saved successfully.")
