import pandas as pd

# Input files
csv_file = "all_4cities_vbd3w.csv"
placekeys_file = "test_placekeys.txt"
output_file = "data/all_4cities_vbd3w_train_set.csv"

# Read placekeys (one per line, no header in txt)
with open(placekeys_file, "r") as f:
    placekeys = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(placekeys)} placekeys to exclude")

# Load CSV
df = pd.read_csv(csv_file)
print(f"Original CSV has {len(df)} rows")

# Filter rows where placekey column does NOT match (using ~)
filtered = df[~df["placekey"].isin(placekeys)]

# Save filtered data
filtered.to_csv(output_file, index=False)

print(f"Training set: {len(filtered)} rows out of {len(df)} (excluded {len(df) - len(filtered)} rows)")
print(f"Saved to {output_file}")

# Verification
excluded_placekeys = set(df["placekey"]) & set(placekeys)
print(f"Verification: {len(excluded_placekeys)} unique placekeys were excluded")
