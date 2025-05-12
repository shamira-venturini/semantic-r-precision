from semantic_r_precision import calculate_sem_r_p


print("--- Example 1: No Exact Match, Good Semantic Match ---")
preds1 = ["wavelet based blind watermarking", "wavelet domain transformations", "wavelet filter parametrization"]
refs1 = ["blind watermarking", "multiple watermarking", "wavelet packets", "parameterized wavelet filters"]
score1 = calculate_sem_r_p(preds1, refs1, k=3)
print(f"Predictions: {preds1}")
print(f"References: {refs1}")
print(f"SemR-p Score: {score1:.4f}\n")


print("--- Example 2: Exact Matches ---")
preds2 = ["earthquakes", "yellowstone national park"]
refs2 = ["earthquakes", "yellowstone national park"]
score2 = calculate_sem_r_p(preds2, refs2, k=3)
print(f"Predictions: {preds2}")
print(f"References: {refs2}")
print(f"SemR-p Score: {score2:.4f}\n")


print("--- Example 3: SemR-p Penalises Poor Top Rank")
preds3 = [
    "system design",
    "data architecture",
    "temporal logic",
    "multi agent systems"
]
refs3 = [
    "temporal logic",
    "multi agent systems",
    "model checking"
]

score3 = calculate_sem_r_p(preds3, refs3, k=3)
print(f"Predictions: {preds3}")
print(f"References: {refs3}")
print(f"SemR-p Score: {score3:.4f}\n")
