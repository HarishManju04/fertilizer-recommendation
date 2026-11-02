# Fertilizer Recommendation System

A Flask web application that recommends a fertilizer based on soil and crop details, nutrient levels, and environmental conditions. The application trains a scikit-learn decision tree from `fertilizer_recommendation.csv` when it starts.

## Requirements

- Python 3.8 or newer
- pip

## Setup

From the project directory, create and activate a virtual environment in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the application dependencies:

```powershell
python -m pip install Flask pandas numpy scikit-learn
```

## Run

Start the Flask development server:

```powershell
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. The CSV file must remain in the project directory alongside `app.py`.

## Using the application

Enter the farm's nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, and soil moisture values, then select a soil type and crop type. Submit the form to see the predicted fertilizer.

Soil and crop choices are learned from the CSV data. The current dataset includes these soil types: Sandy, Loamy, Black, Red, and Clayey; and these crops: Maize, Sugarcane, Cotton, Tobacco, Paddy, Barley, Wheat, Millets, Oil seeds, Pulses, and Ground Nuts.

## Project files

- `app.py` — loads and prepares the dataset, trains the decision tree, and serves the web application.
- `fertilizer_recommendation.csv` — training data used to build the model at startup.
- `templates/index.html` — web form and recommendation display.
- `FertilizerRecommendation.ipynb` — Jupyter notebook for exploring the recommendation dataset.

## Notes

The model is trained from the included dataset each time the app starts. Recommendations reflect that dataset and are informational; verify fertilizer decisions with appropriate soil testing and local agricultural guidance.
