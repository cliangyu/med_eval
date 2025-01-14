import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Generate subcellular distribution plots')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input JSONL file')
    parser.add_argument('--output', type=str, default='subcellular_distribution.png',
                      help='Output plot file path (default: subcellular_distribution.png)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create parent directory for output file if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read JSONL file into a list of dictionaries
    data = []
    with open(args.input, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Explode the localizations column since it's a list
    df_exploded = df.explode('localizations')

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot cell line distribution
    cell_line_counts = df['cell_line'].value_counts()
    cell_line_counts.plot(kind='bar', ax=ax1)
    ax1.set_title('Distribution of Cell Lines')
    ax1.set_xlabel('Cell Line')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)

    # Plot localization distribution
    localization_counts = df_exploded['localizations'].value_counts()
    localization_counts.plot(kind='bar', ax=ax2)
    ax2.set_title('Distribution of Subcellular Localizations')
    ax2.set_xlabel('Localization')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Print numerical statistics and save location
    print("\nCell Line Distribution:")
    print(cell_line_counts)
    print("\nLocalization Distribution:")
    print(localization_counts)
    print(f"\nPlot saved to: {output_path}")

if __name__ == '__main__':
    main()