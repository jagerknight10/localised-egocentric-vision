from egovision.features.cache import load_feature_cache

full = load_feature_cache("data/features_paired_full/S1_Cheese_C1.pt")
restricted = load_feature_cache("data/features_restricted/S1_Cheese_C1.pt")

print("full:", full["features"].shape)
print("restricted:", restricted["features"].shape)
print("same frames:", (full["frame_indices"] == restricted["frame_indices"]).all().item())
print("same labels:", (full["labels"] == restricted["labels"]).all().item())

