import pandas as pd
import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Cap samples per cell line and localization')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input JSONL file')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to output JSONL file')
    parser.add_argument('--cap', type=int, default=5000,
                      help='Maximum number of samples per cell line (default: 5000)')
    parser.add_argument('--loc_cap', type=int, default=1000,
                      help='Maximum number of samples per localization (default: 2000)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility (default: 42)')
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

    # Print original distributions
    print("\nOriginal cell line distribution:")
    print(df['cell_line'].value_counts())
    print("\nOriginal localization distribution:")
    print(df['localizations'].explode().value_counts())

    # Cap each cell line to specified number of samples
    df = df.groupby('cell_line').apply(
        lambda x: x.sample(n=min(len(x), args.cap), random_state=args.seed)
    ).reset_index(drop=True)

    # Cap each localization type to specified number of samples
    # First, create a DataFrame with exploded localizations
    df_exploded = df.explode('localizations')
    
    # Sample for each localization type using loc_cap instead of cap
    df_exploded = df_exploded.groupby('localizations').apply(
        lambda x: x.sample(n=min(len(x), args.loc_cap), random_state=args.seed)
    ).reset_index(drop=True)
    
    # Remove duplicates that might have been created due to multiple localizations
    df = df_exploded.drop_duplicates(subset=['images']).copy()

    # Print new distributions
    print("\nNew cell line distribution after capping:")
    print(df['cell_line'].value_counts())
    print("\nNew localization distribution after capping:")
    print(df['localizations'].explode().value_counts())

    # Save to JSONL
    with open(output_path, 'w') as f:
        for _, row in df.iterrows():
            json_line = row.to_dict()
            f.write(json.dumps(json_line) + '\n')

    print(f"\nSaved capped dataset to: {output_path}")
    print(f"Total samples in output: {len(df)}")

if __name__ == '__main__':
    main()