import argparse
parser = argparse.ArgumentParser(description="Dataset processing script")
parser.add_argument('--dataset-id', required=True, type=int, help="dataset id to process and store")
args = parser.parse_args()

if args.dataset_id == 1:
    print(1)

elif args.dataset_id == 2:
    print(2)

else:
    print(3)

# Load data

# Segment data

# Shuffle data

# Save data
