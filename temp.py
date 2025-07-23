import pickle

with open("data/lst0.pkl", "rb") as f:
    loaded_list = pickle.load(f)

print(loaded_list)  # 출력: [1, 2, 3, 4, 5]