import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import vaex

print("Loading data from CSV...")
df = vaex.from_csv('/home/ly/d/data/pubmed_landscape_data_2024_v2.csv', 
                   usecols=['x', 'y', 'Colors', 'Labels', 'PMID'])

# Load PMCIDs list
pmcids_int = np.load('/home/ly/d/data/unique_pmcids_int.npy')
print(f"Loaded {len(pmcids_int):,} unique PMCIDs")

print(f"Total rows in dataset: {len(df):,}")
sample_size = 1_000_000

# plot unique values of Labels
print(df['Labels'].unique())

# First filter for labeled points
labeled_mask = df.Labels != "unlabeled"
labeled_df = df[labeled_mask]
print(f"Number of labeled points: {len(labeled_df):,}")

# Filter for points in BIOMEDICA list biomedica_pmcs.npy
biomedica_pmcids = np.load('/home/ly/d/data/biomedica_pmcs.npy')
biomedica_mask = df.PMID.isin(biomedica_pmcids)
biomedica_df = df[biomedica_mask]
print(f"Number of points in BIOMEDICA list: {len(biomedica_df):,}")

# Sample from labeled points first, then fill remaining with unlabeled if needed
if len(labeled_df) >= sample_size:
    sampled_df = labeled_df.sample(n=sample_size)
else:
    # Take all labeled points
    unlabeled_sample_size = sample_size - len(labeled_df)
    unlabeled_df = df[~labeled_mask].sample(n=unlabeled_sample_size)
    sampled_df = vaex.concat([labeled_df, unlabeled_df])

# Convert to pandas for easier plotting
pdf = sampled_df.to_pandas_df()
print(f"Final sample size: {len(pdf):,}")

print("Creating plot...")
# Create figure with white background
plt.figure(figsize=(12, 12), dpi=100, facecolor='white')
plt.style.use('default')

# Helper function to validate and fix colors
def clean_color(color):
    try:
        rgb = to_rgb(color)
        # Mute the color by mixing with gray
        muted = tuple(0.7 * c + 0.3 * 0.8 for c in rgb)  # 0.8 is a light gray value
        return muted
    except:
        return (0.8, 0.8, 0.8)  # Light gray as default

print("Cleaning colors...")
colors = []
pmids = pdf['PMID'].values
for i, c in enumerate(pdf['Colors']):
    if pmids[i] in pmcids_int:
        colors.append('black')
    else:
        colors.append(clean_color(c))

print(f"Number of points colored black (in PMCID list): {colors.count('black'):,}")

print("Creating scatter plot...")
sizes = [10 if pmid in pmcids_int else 1 for pmid in pmids]
plt.scatter(pdf['x'], pdf['y'], c=colors, s=sizes, alpha=0.5)  # reduced alpha for less contrast

print("Adding labels...")
# Get one point for each unique label, excluding 'unlabeled'
unique_labels = pdf[pdf['Labels'] != 'unlabeled']['Labels'].unique()
labeled_points = []
for label in unique_labels:
    # Get all points with this label
    label_group = pdf[pdf['Labels'] == label]
    # Select point closest to the center of its group
    center_x = label_group['x'].mean()
    center_y = label_group['y'].mean()
    closest_point = label_group.iloc[((label_group['x'] - center_x)**2 + 
                                    (label_group['y'] - center_y)**2).argmin()]
    labeled_points.append(closest_point)

# Convert to DataFrame for consistent processing
labeled_points = pd.DataFrame(labeled_points)

# Add labels with overlap prevention
used_positions = []
for _, point in labeled_points.iterrows():
    x, y = point['x'], point['y']
    label = point['Labels']
    
    # Find position with least overlap
    best_pos = (5, 5)  # default offset
    min_overlap = float('inf')
    
    for dx in [-20, -10, 0, 10, 20]:
        for dy in [-20, -10, 0, 10, 20]:
            overlap = sum((abs(x + dx - px) < 40 and abs(y + dy - py) < 20) 
                         for px, py in used_positions)
            if overlap < min_overlap:
                min_overlap = overlap
                best_pos = (dx, dy)
    
    plt.annotate(label,
                (x, y),
                xytext=best_pos, textcoords='offset points',
                fontsize=16, alpha=0.7,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    used_positions.append((x + best_pos[0], y + best_pos[1]))

plt.grid(True, alpha=0.2)
plt.xlim(-220, 220)
plt.ylim(-220, 220)

# Hide axes
plt.axis('off')

print("Saving plot...")
plt.savefig('data_visualization.jpg', format='jpg',
            bbox_inches='tight', dpi=100,
            facecolor='white')
plt.savefig('data_visualization.pdf', format='pdf', 
            bbox_inches='tight', dpi=100, 
            facecolor='white')
plt.close()
print("Done!")