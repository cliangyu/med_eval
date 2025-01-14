import numpy as np

# Load the data
pmcids = np.load('/home/ly/d/data/unique_pmcids.npy')

# Remove 'PMC' prefix and convert to integers
pmcids_int = np.array([int(id_[3:]) for id_ in pmcids])

# Save the converted array
np.save('/home/ly/d/data/unique_pmcids_int.npy', pmcids_int)

# Verify the conversion
print(f"Total number of PMCIDs: {len(pmcids_int):,}")
print(f"Data type: {pmcids_int.dtype}")
print("\nFirst 10 PMCIDs:")
print(pmcids_int[:10])
print(f"\nNumber of unique PMCIDs: {len(np.unique(pmcids_int)):,}")
print(f"\nValue range: {pmcids_int.min():,} to {pmcids_int.max():,}")