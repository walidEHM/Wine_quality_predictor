from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import traceback

app = Flask(__name__)

# ---------- Chargement des objets ----------
try:
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print(f"✅ Modèle chargé : attend {model.n_features_in_} features")
except Exception as e:
    print(f"❌ Erreur lors du chargement des fichiers : {e}")
    raise

# Ordre des 11 colonnes (identique à celui du dataset d'entraînement)
FEATURE_NAMES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]

# Labels optionnels pour un affichage lisible
QUALITY_LABELS = {
    3: "Qualité 3 – Médiocre",
    4: "Qualité 4 – Passable",
    5: "Qualité 5 – Moyen",
    6: "Qualité 6 – Bon",
    7: "Qualité 7 – Très bon",
    8: "Qualité 8 – Excellent"
}

@app.route("/")
def home():
    """Page d'accueil avec le formulaire."""
    return render_template("index.html", feature_names=FEATURE_NAMES)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Reçoit les 11 valeurs du formulaire, applique uniquement le scaling,
    puis retourne la prédiction et les probabilités.
    """
    try:
        # Récupération des données du formulaire
        input_dict = {}
        for feat in FEATURE_NAMES:
            val = request.form.get(feat)
            if val is None or val.strip() == "":
                return jsonify({"error": f"Le champ '{feat}' est vide"}), 400
            input_dict[feat] = float(val)

        # Création d'un DataFrame (1 ligne, 11 colonnes)
        input_df = pd.DataFrame([input_dict], columns=FEATURE_NAMES)

        # Scaling
        input_scaled = scaler.transform(input_df)

        # Prédiction
        pred = int(model.predict(input_scaled)[0])
        proba = model.predict_proba(input_scaled)[0]
        classes = model.classes_.astype(int).tolist()

        # Probabilités formatées
        proba_dict = {c: round(float(p), 4) for c, p in zip(classes, proba)}

        result = {
            "prediction": pred,
            "label": QUALITY_LABELS.get(pred, f"Qualité {pred}"),
            "probabilities": proba_dict
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)