from flask import Flask, request, render_template
import os, re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__, template_folder="templates")

def normalize(s):
    return re.sub(r'[^a-z0-9]+', ' ', str(s).lower()).strip()

def find_column_by_tokens(columns, tokens):
    cols_norm = {normalize(c): c for c in columns}
    # 1) try exact normalized token match or substring match
    for tok in tokens:
        for norm, orig in cols_norm.items():
            if tok in norm:
                return orig
    # 2) try relaxed matching: token prefix
    for tok in tokens:
        for norm, orig in cols_norm.items():
            if norm.startswith(tok) or norm.endswith(tok):
                return orig
    # 3) try exact original match (case-insensitive)
    for tok in tokens:
        for orig in columns:
            if tok.lower() == orig.lower():
                return orig
    return None

# load CSV (path relative to this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "fertilizer_recommendation.csv")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print("CSV loaded. Columns:", list(df.columns))

# robust token lists including common misspellings / variants
candidates = {
    'Temperature':   ['temperature','temp','temparature','t','tempr'],
    'Humidity':      ['humidity','hum','h'],
    'Moisture':      ['moisture','soil moisture','soilmoisture','moist','soil','sm'],
    'Soil Type':     ['soil type','soiltype','soil','soil type'],
    'Crop Type':     ['crop type','croptype','crop'],
    'Nitrogen':      ['nitrogen','nitro','n'],
    'Phosphorous':   ['phosphorous','phosphorus','phosphor','p'],
    'Potassium':     ['potassium','potass','k'],
    'Fertilizer':    ['fertilizer name','fertilizer','fertiliser','fertilizername','fertilizer name','fertil','fertiliser name','fertilizer_name']
}

mapping = {}
for canon, toks in candidates.items():
    found = find_column_by_tokens(df.columns, toks)
    mapping[canon] = found

missing = [k for k,v in mapping.items() if v is None]
if missing:
    raise KeyError(f"Required column(s) not found in CSV: {missing}. Detected columns: {list(df.columns)}")

# rename to canonical names
df = df.rename(columns={v: k for k, v in mapping.items()})
print("Column mapping:", mapping)

# coerce numeric columns and drop invalid rows
for col in ['Temperature','Humidity','Moisture','Nitrogen','Phosphorous','Potassium']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['Temperature','Humidity','Moisture','Nitrogen','Phosphorous','Potassium','Soil Type','Crop Type','Fertilizer'])

# encode categorical features
le_soil = LabelEncoder()
le_crop = LabelEncoder()
df['Soil Type'] = le_soil.fit_transform(df['Soil Type'].astype(str))
df['Crop Type'] = le_crop.fit_transform(df['Crop Type'].astype(str))

# explicit feature order (must match prediction order below)
FEATURE_ORDER = ['Temperature','Humidity','Moisture','Soil Type','Crop Type','Nitrogen','Potassium','Phosphorous']
X = df[FEATURE_ORDER]
y = df['Fertilizer']

print("Training set shape:", X.shape, "Classes:", y.nunique(), "Class counts:", y.value_counts().to_dict())

# train decision tree
dtc = DecisionTreeClassifier(random_state=0, max_depth=12)
dtc.fit(X, y)
print("Model trained.")

soil_map = {str(c).lower(): int(le_soil.transform([c])[0]) for c in le_soil.classes_}
crop_map = {str(c).lower(): int(le_crop.transform([c])[0]) for c in le_crop.classes_}
print("Soil classes:", list(soil_map.keys()))
print("Crop classes:", list(crop_map.keys()))

@app.route('/', methods=['GET','POST'])
def index():
    fertilizer = None
    error = None
    if request.method == 'POST':
        try:
            # read and coerce inputs
            n = float(request.form.get('n', '0') or 0)
            p = float(request.form.get('p', '0') or 0)
            k = float(request.form.get('k', '0') or 0)
            t = float(request.form.get('t', '0') or 0)
            h = float(request.form.get('h', '0') or 0)
            sm = float(request.form.get('soilMoisture', '0') or 0)
            soil = (request.form.get('soil') or '').strip()
            crop = (request.form.get('crop') or '').strip()

            if not soil or not crop:
                error = "Select Soil and Crop"
            else:
                soil_enc = soil_map.get(soil.lower())
                crop_enc = crop_map.get(crop.lower())
                if soil_enc is None or crop_enc is None:
                    error = "Soil/Crop not recognized. See server log for allowed values."
                else:
                    user_input = np.array([[t, h, sm, soil_enc, crop_enc, n, k, p]], dtype=float)
                    print("User input (ordered):", user_input.tolist())
                    fertilizer = dtc.predict(user_input)[0]
                    print("Prediction:", fertilizer)
        except Exception as e:
            error = str(e)
    return render_template('index.html', fertilizer=fertilizer, error=error)

if __name__ == '__main__':
    app.run(debug=True)