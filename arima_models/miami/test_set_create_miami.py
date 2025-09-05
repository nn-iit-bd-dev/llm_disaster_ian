import pandas as pd

# Input files
csv_file = "Miami_vbd3w.csv"
placekeys_file = "miami_test.placekeys.txt"
output_file = "miami_vbd3w_for_test.csv"

# Read placekeys (one per line, no header in txt)
with open(placekeys_file, "r") as f:
    placekeys = [line.strip() for line in f if line.strip()]

# Load CSV
df = pd.read_csv(csv_file)

# Filter rows where placekey column matches
filtered = df[df["placekey"].isin(placekeys)]

# Save filtered data
filtered.to_csv(output_file, index=False)

print(f"Filtered {len(filtered)} rows out of {len(df)}. Saved to {output_file}")



