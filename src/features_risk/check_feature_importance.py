import xgboost as xgb
from severity_score import build_training_data

X, y, feature_names = build_training_data()
model = xgb.XGBClassifier()
model.load_model("model/severity_model.json")

print("=== Feature Importance ===")
for name, imp in sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1]):
    print(f"{name}: {imp:.4f}")