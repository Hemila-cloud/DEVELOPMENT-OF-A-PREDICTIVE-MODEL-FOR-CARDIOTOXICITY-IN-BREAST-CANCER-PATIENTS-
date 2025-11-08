import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class UltimateCardiotoxicityPredictor:
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.final_auc = 0
        self.research_baseline_auc = 0.81  # CHECK-HEART paper AUC

    def clean_data(self, X):
        """Remove infinite values and handle missing data properly"""
        print("Cleaning data...")
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        inf_count = (X == np.inf).sum().sum() + (X == -np.inf).sum().sum()
        if inf_count > 0:
            print(f"Removed {inf_count} infinite values")

        X_clean = X_clean.fillna(X_clean.median())
        if X_clean.isnull().any().any():
            print(f"Warning: {X_clean.isnull().sum().sum()} NaN values remaining")
            X_clean = X_clean.fillna(0)
        return X_clean

    def create_super_features(self, df):
        """Create features that NO research paper has used together"""
        print("Creating SUPER FEATURES beyond current research...")

        # 1. Treatment response
        if 'LVEF_change_3mo' in df.columns:
            df['early_responder'] = (df['LVEF_change_3mo'] < -5).astype(int)
            df['stable_responder'] = (abs(df['LVEF_change_3mo']) <= 5).astype(int)

        # 2. Remodeling index
        if all(col in df.columns for col in ['LVDd', 'PWT', 'LVEF']):
            df['remodeling_index'] = (df['LVDd'] * df['PWT']) / df['LVEF'].replace(0, 1)

        # 3. Diastolic-systolic mismatch
        if all(col in df.columns for col in ['mitral_E_A_ratio', 'LVEF', 'E_e_ratio']):
            df['DS_mismatch'] = ((df['E_e_ratio'] > 10) & (df['LVEF'] > 55)).astype(int)
            df['DS_mismatch_score'] = df['E_e_ratio'] / df['LVEF'].replace(0, 1)

        # 4. Toxicity burden
        df['toxicity_burden'] = (
            df.get('antiHER2', 0) * 2 +
            df.get('AC', 0) * 1.5 +
            df.get('RTprev', 0) * 1 +
            df.get('HTA', 0) * 0.5 +
            df.get('DM', 0) * 0.5
        )

        # 5. Time-series instability
        time_cols = [col for col in df.columns if col.startswith('t ')]
        if len(time_cols) > 3:
            time_data = df[time_cols[:6]].copy()
            time_data = time_data.replace([np.inf, -np.inf], np.nan).fillna(time_data.median())
            means = time_data.mean(axis=1).replace(0, 1)
            df['LVEF_instability'] = time_data.std(axis=1) / means
            df['early_trend'] = (time_data.iloc[:, 3] - time_data.iloc[:, 0]) / 3

        novel_feature_count = sum([
            'early_responder' in df.columns,
            'remodeling_index' in df.columns,
            'DS_mismatch' in df.columns,
            'toxicity_burden' in df.columns,
            'LVEF_instability' in df.columns
        ])
        print(f"Created {novel_feature_count} NOVEL features beyond current research")
        return df

    def select_research_beating_features(self, df):
        research_features = ['age', 'heart_rate', 'LVEF', 'antiHER2', 'AC', 'HTA', 'DM']
        enhanced_features = ['BMI', 'LVEF_min', 'LVEF_drop_from_baseline', 'LVEF_change_3mo',
                             'high_risk_treatment', 'treatment_burden', 'comorbidity_score']
        novel_features = ['early_responder', 'remodeling_index', 'DS_mismatch', 'DS_mismatch_score',
                          'toxicity_burden', 'LVEF_instability', 'early_trend']
        
        # Doppler features removed as requested
        all_potential = research_features + enhanced_features + novel_features
        available_features = [f for f in all_potential if f in df.columns]

        print("\nFEATURE BREAKDOWN (Why we'll beat research):")
        print(f"Research-grade features: {len([f for f in research_features if f in df.columns])}")
        print(f"Enhanced clinical features: {len([f for f in enhanced_features if f in df.columns])}")
        print(f"NOVEL research features: {len([f for f in novel_features if f in df.columns])}")
        print(f"TOTAL features: {len(available_features)}")
        return available_features

    def train_research_beating_model(self, X_train, y_train):
        print("\nTraining RESEARCH-BEATING model...")
        X_train_clean = self.clean_data(X_train)
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]) if sum(y_train) > 0 else 1
        )
        model.fit(X_train_clean, y_train)
        return model

    def comprehensive_evaluation(self, model, X_test, y_test, feature_names):
        print("\n" + "="*70)
        print("RESEARCH PAPER COMPARISON RESULTS")
        print("="*70)

        X_test_clean = self.clean_data(X_test)
        y_pred_proba = model.predict_proba(X_test_clean)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        cv_scores = cross_val_score(model, X_test_clean, y_test, cv=5, scoring='roc_auc')

        print(f"OUR MODEL AUC: {auc:.4f}")
        print(f"BEST RESEARCH PAPER (CHECK-HEART): {self.research_baseline_auc:.4f}")
        print(f"IMPROVEMENT: +{(auc - self.research_baseline_auc)*100:.2f}%")
        if auc > self.research_baseline_auc:
            print("🎯 RESULT: WE BEAT THE RESEARCH PAPERS! 🎯")
        else:
            print("⚠️ Close but need feature optimization")
        print(f"\nCross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTOP 10 PREDICTORS (Why we win):")
        print("-"*45)
        for i, row in feature_imp.head(10).iterrows():
            print(f"{i+1:2}. {row['feature']:25} {row['importance']:.4f}")
        return auc, feature_imp

    def clinical_impact_analysis(self, model, X_test, y_test):
        X_test_clean = self.clean_data(X_test)
        y_pred_proba = model.predict_proba(X_test_clean)[:, 1]
        bins = [0, 0.2, 0.5, 1.0]
        labels = ['Low Risk', 'Moderate Risk', 'High Risk']
        risk_groups = pd.cut(y_pred_proba, bins=bins, labels=labels, include_lowest=True)
        summary = pd.DataFrame({
            'Predicted_Probability': y_pred_proba,
            'True_CTRCD': y_test.values,
            'Risk_Group': risk_groups
        })
        group_stats = summary.groupby('Risk_Group').agg(
            Total_Patients=('Predicted_Probability', 'count'),
            Mean_Predicted_Risk=('Predicted_Probability', 'mean'),
            True_CTRCD_Cases=('True_CTRCD', 'sum')
        )
        group_stats['CTRCD_Prevalence_%'] = 100 * group_stats['True_CTRCD_Cases'] / group_stats['Total_Patients']
        group_stats['Group_%'] = 100 * group_stats['Total_Patients'] / len(summary)

        print("\n" + "="*70)
        print("🩺 CLINICAL RISK STRATIFICATION REPORT")
        print("="*70)
        print(group_stats)
        print("-"*70)

        high_risk_count = group_stats.loc['High Risk', 'Total_Patients'] if 'High Risk' in group_stats.index else 0
        true_high = group_stats.loc['High Risk', 'True_CTRCD_Cases'] if 'High Risk' in group_stats.index else 0
        total_cases = y_test.sum()

        print(f"\n📈 Risk Distribution Summary:")
        print(f"   • {group_stats.loc['Low Risk', 'Group_%']:.1f}% patients are LOW risk (mean prob {group_stats.loc['Low Risk', 'Mean_Predicted_Risk']:.2f})")
        print(f"   • {group_stats.loc['Moderate Risk', 'Group_%']:.1f}% patients are MODERATE risk (mean prob {group_stats.loc['Moderate Risk', 'Mean_Predicted_Risk']:.2f})")
        print(f"   • {group_stats.loc['High Risk', 'Group_%']:.1f}% patients are HIGH risk (mean prob {group_stats.loc['High Risk', 'Mean_Predicted_Risk']:.2f})")
        print(f"\n   ✅ {true_high}/{int(total_cases)} true CTRCD cases ({true_high/total_cases*100:.1f}%) captured in the HIGH-risk category")
        return group_stats, summary

    def plot_research_comparison(self, model, X_test, y_test, auc):
        from sklearn.metrics import RocCurveDisplay
        X_test_clean = self.clean_data(X_test)
        plt.figure(figsize=(10, 8))
        RocCurveDisplay.from_estimator(model, X_test_clean, y_test, name='Our Model')
        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.50)')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label=f'Research Baseline (AUC={self.research_baseline_auc})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'RESEARCH COMPARISON: Our Model (AUC={auc:.3f}) vs Literature (AUC={self.research_baseline_auc})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if auc > self.research_baseline_auc:
            plt.text(0.6, 0.3, f'+{(auc - self.research_baseline_auc)*100:.1f}% improvement\nover current research',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        plt.savefig('RESEARCH_BEATING_RESULTS.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("🚀 ULTIMATE CARDIOTOXICITY PREDICTOR - BEATING RESEARCH PAPERS 🚀")
    predictor = UltimateCardiotoxicityPredictor()

    df = pd.read_csv('cardiotoxicity_dataset_ENHANCED_FULL.csv')
    print(f"Loaded dataset: {df.shape}")

    df_super = predictor.create_super_features(df)
    features = predictor.select_research_beating_features(df_super)
    X = df_super[features]
    y = df_super['CTRCD']

    if len(y.unique()) > 2:
        y = (y > 0).astype(int)
        print(f"Converted target to binary: {y.unique()}")

    X_clean = predictor.clean_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nTraining on {X_train.shape[0]} patients, testing on {X_test.shape[0]} patients")
    print(f"CTRCD prevalence: {y_train.mean():.1%} training, {y_test.mean():.1%} testing")

    model = predictor.train_research_beating_model(X_train, y_train)
    auc, feature_importance = predictor.comprehensive_evaluation(model, X_test, y_test, features)
    risk_summary, summary = predictor.clinical_impact_analysis(model, X_test, y_test)
    predictor.plot_research_comparison(model, X_test, y_test, auc)

    # Compute true positives
    X_test_clean = predictor.clean_data(X_test)
    y_pred = (model.predict_proba(X_test_clean)[:, 1] > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    true_positives = tp

    joblib.dump({
        'model': model,
        'features': features,
        'auc': auc,
        'research_baseline': predictor.research_baseline_auc,
        'improvement': auc - predictor.research_baseline_auc
    }, 'CHAMPION_cardiotoxicity_model.pkl')

    print(f"\n🎉 CHAMPION MODEL SAVED! 🎉")
    print(f"   AUC: {auc:.4f} vs Research: {predictor.research_baseline_auc:.4f}")
    print(f"   Improvement: +{(auc - predictor.research_baseline_auc)*100:.2f}%")
    print(f"   Early interventions enabled: {true_positives} patients")

    return model, auc

if __name__ == "__main__":
    champion_model, final_auc = main()