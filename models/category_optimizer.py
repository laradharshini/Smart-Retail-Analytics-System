import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

MODEL_PATH = "models/category_trend_model.pkl"
DATA_PATH = "Customer Experience Dataset.csv"

def train_category_model():
    print("🔎 Starting training routine...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found at: {DATA_PATH}")
        return

    try:
        df = pd.read_csv(DATA_PATH, encoding='latin1')
        print(f"✅ Loaded dataset with shape: {df.shape}")
    except Exception as e:
        print("❌ Failed to load dataset:", str(e))
        return

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )

    column_map = {
        'which_product_categories_do_you_frequently_buy_select_all_that_apply': 'frequent_categories',
        'which_categories_do_you_prefer_to_buy_during_offers_select_all_that_apply': 'offer_categories',
        'which_offer_types_do_you_prefer__select_all_that_apply': 'offer_types',
        'do_you_consider_ecofriendliness_when_choosing_a_product': 'eco_friendly',
        'do_you_use_any_price_comparison_or_shopping_apps_before_buying': 'price_comparison'
    }

    missing_cols = [col for col in column_map if col not in df.columns]
    if missing_cols:
        print(f"❌ Required columns not found: {missing_cols}")
        print("Available columns (sample):", list(df.columns)[:30])
        return

    df.rename(columns=column_map, inplace=True)
    print("✅ Columns cleaned and renamed successfully.")

    def parse_input(val):
        if isinstance(val, str):
            # support '|' separators also (sometimes CSVs use pipes)
            parts = [v.strip() for v in val.replace("|", ",").split(",") if v.strip()]
            return parts
        elif isinstance(val, list):
            return val
        return []

    # Avoid dropping rows too aggressively — report how many rows will be removed
    pre_rows = df.shape[0]
    df.dropna(subset=list(column_map.values()), inplace=True)
    post_rows = df.shape[0]
    print(f"➡️ Rows before dropna: {pre_rows}, after dropna: {post_rows}")
    if post_rows == 0:
        print("❌ No rows left after dropna. Check CSV for empty strings or different column values.")
        return

    df['frequent_categories'] = df['frequent_categories'].apply(parse_input)
    df['offer_categories'] = df['offer_categories'].apply(parse_input)
    df['offer_types'] = df['offer_types'].apply(parse_input)

    # Fit MultiLabelBinarizers but guard against empty class lists
    mlb_freq = MultiLabelBinarizer()
    mlb_offer = MultiLabelBinarizer()
    mlb_type = MultiLabelBinarizer()

    try:
        X_freq = mlb_freq.fit_transform(df['frequent_categories'])
        X_offer = mlb_offer.fit_transform(df['offer_categories'])
        X_type = mlb_type.fit_transform(df['offer_types'])
    except Exception as e:
        print("❌ Error while binarizing multilabel columns:", str(e))
        return

    # Make eco_friendly numeric (yes->1, no->0). If value unknown, set 0 (or choose 0.5 as missing)
    df['eco_friendly'] = (
        df['eco_friendly']
        .astype(str)
        .str.strip()
        .str.lower()
        .map({'yes': 1, 'no': 0})
        .fillna(0)
        .astype(int)
    )

    # Build final feature DataFrame
    X_parts = []
    if X_freq.size:
        X_parts.append(pd.DataFrame(X_freq, columns=["freq_" + c for c in mlb_freq.classes_]))
    if X_offer.size:
        X_parts.append(pd.DataFrame(X_offer, columns=["offer_" + c for c in mlb_offer.classes_]))
    if X_type.size:
        X_parts.append(pd.DataFrame(X_type, columns=["type_" + c for c in mlb_type.classes_]))

    # eco_friendly column as named Series
    X_parts.append(df['eco_friendly'].reset_index(drop=True).rename("eco_friendly"))

    X_final = pd.concat(X_parts, axis=1)
    print(f"✅ Final feature matrix shape: {X_final.shape}")

    # Example binary target: is the respondent a snack buyer?
    def is_snack_buyer(cats):
        if not isinstance(cats, list):
            return 0
        lowered = [c.lower() for c in cats]
        return int(any('snack' in c for c in lowered))  # matches 'snacks' or 'snack items'

    y = df['frequent_categories'].apply(is_snack_buyer)

    # Check target distribution
    print("Target distribution (value:count):")
    print(y.value_counts().to_dict())

    if y.sum() == 0:
        print("❌ No positive examples for snack buyers in the data. The model won't learn anything.")
        # Still proceed if you want, but warn:
        # return

    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    bundle = {
        "model": clf,
        "mlb_freq": mlb_freq,
        "mlb_offer": mlb_offer,
        "mlb_type": mlb_type,
        "features": list(X_final.columns),
        "target": "frequent_snack_buyer"
    }

    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(bundle, f)
        print(f"✅ Model and components saved to: {MODEL_PATH}")
        # Print a quick test score
        score = clf.score(X_test, y_test)
        print(f"📊 Model test accuracy: {score:.3f}")
    except Exception as e:
        print("❌ Failed to save model:", str(e))
        return

def generate_category_insights(user_inputs):
    insights = []

    if user_inputs.get("frequent_categories"):
        frequent = user_inputs["frequent_categories"]
        insights.append(f"You frequently purchase: {', '.join(frequent)}.")
        if any(x.lower() == "snacks" for x in frequent) or any(x.lower() == "beverages" for x in frequent):
            insights.append("To save on frequent items like Snacks or Beverages, consider bulk buying or subscriptions.")
        if any(x.lower() == "personal care" for x in frequent):
            insights.append("Combo packs or loyalty programs could help you save in Personal Care.")

    if user_inputs.get("offer_categories"):
        offer_cats = user_inputs["offer_categories"]
        insights.append(f"You look for offers in: {', '.join(offer_cats)}.")
        if len(offer_cats) >= 3:
            insights.append("You explore many deal categories—try smart carts to combine offers and save more.")

    offer_types = user_inputs.get("offer_types", [])
    if offer_types:
        insights.append(f"Preferred offer types: {', '.join(offer_types)}.")
        if any("buy 1 get 1" in ot.lower() for ot in offer_types):
            insights.append("BOGO offers are great for Snacks or high-use products—keep an eye out.")

    if user_inputs.get("price_comparison", "").lower() in ("yes", "y", "true"):
        insights.append("You compare prices already—enable price alerts for favorite items.")
    else:
        insights.append("Try using price comparison apps to save more on recurring purchases.")

    if user_inputs.get("eco_friendly", "").lower() in ("yes", "y", "true"):
        insights.append("You care about sustainability—try eco-friendly or refillable options.")
    else:
        insights.append("Explore greener options like eco-labeled or refillable products.")

    insights.append("Review your top 3 product categories monthly to improve shopping strategy.")

    return insights

# Simple test usage when script run directly
if __name__ == "__main__":
    # Train the model (this will print reasons if it can't)
    train_category_model()

    # Example usage of insights generator:
    example_input = {
        "frequent_categories": ["Snacks", "Beverages"],
        "offer_categories": ["Snacks", "Household", "Personal Care"],
        "offer_types": ["Buy 1 Get 1 Free", "20% Off"],
        "price_comparison": "Yes",
        "eco_friendly": "No"
    }

    print("\nSample insights for example input:")
    for s in generate_category_insights(example_input):
        print("-", s)
