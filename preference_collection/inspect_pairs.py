import pickle

with open("preference_collection/data/preference_pairs.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Total pairs: {len(data)}")
sample = data[0]
print("Traj A return:", sample["traj_a"]["total_return"])
print("Traj B return:", sample["traj_b"]["total_return"])
print("Preferred:", "A" if sample["preferred"] == 1 else "B")