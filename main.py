from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import io
import shap
import matplotlib.pyplot as plt
import base64

app = FastAPI()

model = joblib.load("churn_pipeline.pkl")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.get("/about",response_class=HTMLResponse)
def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/single", response_class=HTMLResponse)
def single_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.get("/batch", response_class=HTMLResponse)
def batch_page(request: Request):
    return templates.TemplateResponse("batch.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    gender: str = Form(...),
    SeniorCitizen: int = Form(...),
    tenure: int = Form(...),
    MonthlyCharges: float = Form(...)
):
    try:
        # Colonnes attendues par le modèle
        default_data = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 0,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 0.0,
            "TotalCharges": 0.0
        }

        # Remplacer avec les valeurs du formulaire
        default_data.update({
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "tenure": tenure,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": MonthlyCharges * tenure
        })

        # Créer DataFrame
        data_input = pd.DataFrame([default_data])

        # Prédiction
        prediction = model.predict(data_input)[0]
        proba = model.predict_proba(data_input)[0][1] * 100
        result = "❌ Le client va quitter" if prediction == 1 else "✅ Le client reste"

        # Retourner template avec résultat et graphique
        return templates.TemplateResponse(
            "form.html",
            {
                "request": request,
                "result": result,
                "proba": round(proba, 2)
            }
        )

    except Exception as e:
        return HTMLResponse(f"Erreur : {str(e)}", status_code=500)


@app.post("/predict_csv", response_class=HTMLResponse)
async def predict_csv(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Nettoyage
        df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df.dropna(inplace=True)

        # Prédictions
        preds = model.predict(df)
        df["Churn"] = preds

        # Comptage pour graphique
        churn_count = int((df["Churn"] == 1).sum())
        stay_count = int((df["Churn"] == 0).sum())

        # Calcul des pourcentages
        churn_percentage = (churn_count / len(df)) * 100
        stay_percentage = (stay_count / len(df)) * 100

        # Colonnes à afficher (choisis ce que tu veux montrer)
        columns_to_show = [
            "gender",
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Churn"
        ]

        df_display = df[columns_to_show]

        # Tableau HTML propre
        table_html = df_display.to_html(
            classes="min-w-full text-sm text-center border border-gray-300",
            index=False,
            border=0
        )

        return templates.TemplateResponse("upload_csv.html", {
            "request": request,
            "table": table_html,
            "churn_count": churn_count,
            "stay_count": stay_count,
            "churn_percentage": churn_percentage,
            "stay_percentage": stay_percentage
        })

    except Exception as e:
        return HTMLResponse(f"Erreur : {str(e)}", status_code=500)
