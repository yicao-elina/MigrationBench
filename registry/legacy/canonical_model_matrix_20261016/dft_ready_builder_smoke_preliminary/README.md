# DFT-ready builder validation (preliminary)

This directory records a tooling validation of the local periodic-shell builder using one historical prepared candidate. It is not a canonical 639 selection, it is not a DFT result, and it must not be used as evidence that the reduced cell is physically equivalent to the 2050-atom periodic cell.

The builder passed its local gates: finite coordinates and cell, preserved PBC flags, 120 atoms, and minimum interatomic distance above 0.8 Å. The generated QE input retains the QE 7.2 SCF/D3 template and remains `PREPARED_ONLY` until a refreshed accepted-model screen identifies the final DFT candidates.
