import pickle
import random
import os

data_dir = "data"
output_file = os.path.join(data_dir, "2mer-test.pkl")

train_files = [
    "01-at-400-train.pkl",
    "02-ta-400-train.pkl",
    "03-aa-400-train.pkl",
    "04-ca-400-train.pkl"
]

test_samples = []

for f in train_files:
    path = os.path.join(data_dir, f)
    with open(path, "rb") as infile:
        molecules = pickle.load(infile)
        sampled = random.sample(molecules, 2)
        test_samples.extend(sampled)

with open(output_file, "wb") as outfile:
    pickle.dump(test_samples, outfile)

print(f"Created {output_file} with {len(test_samples)} samples.")
