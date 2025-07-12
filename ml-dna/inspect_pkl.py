import pickle
from torch_geometric.data import Data
from utils import get_iso_permuted_dataset

path = "../data/01-at-400-train.pkl"
hhh = "./data/h_s_only_augccpvdz_density.out"
ooo = "./data/o_s_only_augccpvdz_density.out"
ccc = "./data/c_s_only_augccpvdz_density.out"
nnn = "./data/n_s_only_augccpvdz_density.out"
ppp = "./data/p_s_only_augccpvdz_density.out"

dataset = get_iso_permuted_dataset(path, h_iso=hhh, c_iso=ccc, n_iso=nnn, o_iso=ooo, p_iso=ppp)

print(f"Loaded {len(dataset)} samples.")

for i, data in enumerate(dataset):
    print(f"\nSample {i}")
    print(f" - data.pos shape: {data.pos.shape}")     # Atomic positions
    print(f" - data.y shape:   {data.y.shape}")       # Electron density coeffs
    print(f" - atomic numbers: {data.z}")
    if data.y.shape[0] != data.pos.shape[0]:
        print(" ❌ Atom count mismatch! Investigate this one.")
        break
