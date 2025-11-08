# Development of a Predictive Model for Cardiotoxicity in Breast Cancer Patients using Clinical Biomarkers and Time-Series Features**

---

## Project Overview  
This project focuses on developing a **predictive model for early detection of cardiotoxicity** in breast cancer patients undergoing chemotherapy.  
By integrating **clinical biomarkers**, **treatment parameters**, and **echocardiographic time-series data**, the model aims to identify patients at higher risk of developing **cancer therapy–related cardiac dysfunction (CTRCD)** — enabling timely interventions and improved survivorship outcomes.

The primary model used is **XGBoost (Extreme Gradient Boosting)**, optimized for binary classification of high vs low cardiotoxicity risk.  
Comparative benchmarks (e.g., CHECK-HEART, AUC = 0.81) are used for performance reference.  

---

## Key Novelty & Contributions  
- **Hybrid feature design** combining static clinical features and dynamic time-series echocardiographic data.  
- **Novel engineered clinical indices** for enhanced interpretability:
  - `BMI`, `LV_size_ratio`, `wall_thickness_ratio`  
  - `high_risk_treatment` and `treatment_burden` indicators  
  - `comorbidity_score` (combines HTA, DL, DM)  
  - `LVEF_drop_from_baseline`, `LVEF_instability`, `LVEF_change_3mo`  
- **Time-series derived metrics** extracted from LVEF trajectories to capture cardiac function changes over time.  
- **Explainable AI integration** using **LIME** to provide patient-level interpretability for clinical trust.  
- **Automated stratification** into three actionable risk categories:  
  - **Low Risk** → Continue standard therapy  
  - **Moderate Risk** → Monitor closely  
  - **High Risk** → Flag for early cardiac intervention  

---

## Repository Contents  
- `preprocess.py` → Cleans raw data, handles missing values, and prepares training/test splits.  
- `e_data.py` → Performs advanced feature engineering and generates enhanced datasets with clinical + time-series features.  
- `main.py` → Trains the XGBoost model, evaluates results (AUC, F1, accuracy), and visualizes feature importance.  
- `cardiotox.ipynb` → Full Jupyter Notebook with implementation, model evaluation, and visual outputs.  

---

## Algorithm Summary (XGBoost)  
- Builds **a sequence of decision trees** where each tree corrects errors made by previous ones using **gradient boosting**.  
- Uses **binary logistic loss** to generate a probability score between 0–1 for cardiotoxicity risk.  
- Employs **regularization (L1 & L2)** to prevent overfitting and improve generalization.  
- Automatically handles missing data and nonlinear feature interactions.  

**Decision Rule:**  
- High probability → **High-risk patient (needs cardiac monitoring)**  
- Low probability → **Safe continuation of standard therapy**

---

## Explainable AI Integration  
To enhance clinical interpretability, **LIME (Local Interpretable Model-Agnostic Explanations)** was used to visualize the feature contribution for each patient prediction — clarifying why the model classified a patient as high or low risk.

---

## Results Summary  
- **Model:** XGBoost  
- **Precision:** ~0.9836  
- **Recall:** ~1.0  
- **F1-score:** ~0.9917  
- **Key Predictors:** LVEF change, treatment burden, comorbidity score, and heart rate instability.

---

## Tech Stack  
- **Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, xgboost, matplotlib, lime  
- **Environment:** Jupyter Notebook / Google Colab  

---

## Future Scope  
 
- Real-time patient monitoring using streaming cardiac data.  
- Incorporation of imaging-based biomarkers (e.g., MRI or ECG-derived features).  


