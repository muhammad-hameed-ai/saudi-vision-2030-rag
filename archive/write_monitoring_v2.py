import os

monitoring_code = """import os
import pandas as pd
import mlflow
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def run_monitoring_pipeline():
    # 1. Configure local MLflow experiment context
    mlflow.set_experiment("Saudi_Vision_2030_Monitoring")
    
    with mlflow.start_run(run_name="evidently_drift_check"):
        print("Creating baseline and production data tables...")
        
        # 2. Simulate Reference Data (Original Ingestion/Training Style Queries)
        reference_data = pd.DataFrame({
            "user_query": [
                "What are the strategic targets of Saudi Vision 2030?",
                "Explain the clean energy transition goals in the vision document.",
                "How does the kingdom plan to diversify its economic portfolio?",
                "What are the healthcare transformation metrics for 2030?",
                "Tell me about the housing and urban development initiatives."
            ]
        })
        
        # 3. Simulate Production Data (Real incoming traffic)
        production_data = pd.DataFrame({
            "user_query": [
                "Who is playing the football match in Riyadh tonight?",
                "Can you recommend a cheap hotel near the airport?",
                "What is the population size of the kingdom right now?",
                "How to apply for an international tourist e-visa?",
                "Show me the stock pricing charts for oil companies."
            ]
        })
        
        print("Calculating Data Drift Metrics via Evidently AI 0.4.33...")
        
        # 4. Map the text column for the older Evidently engine
        column_mapping = ColumnMapping()
        column_mapping.text_features = ["user_query"]
        
        # 5. Initialize and compile the Metrics Report
        text_report = Report(metrics=[DataDriftPreset()])
        text_report.run(reference_data=reference_data, current_data=production_data, column_mapping=column_mapping)
        
        # 6. Extract calculated parameters safely
        report_dict = text_report.as_dict()
        try:
            drift_share = report_dict["metrics"][0]["result"]["drift_share"]
            drift_detected = report_dict["metrics"][0]["result"]["dataset_drift"]
            
            mlflow.log_metric("drift_share", drift_share)
            mlflow.log_param("drift_detected", str(drift_detected))
            print(f"📊 Tracking complete! Drift Share: {drift_share:.4f} | Drift Detected: {drift_detected}")
        except Exception as e:
            print(f"📊 Tracking completed (exact JSON keys varied): {e}")
            
        # 7. Save structural interactive report as an MLflow Artifact
        os.makedirs("reports", exist_ok=True)
        report_html_path = "reports/text_drift_report.html"
        text_report.save_html(report_html_path)
        
        mlflow.log_artifact(report_html_path, artifact_path="evidently_reports")
        print("✅ HTML Report logged as an interactive MLflow artifact successfully!")

if __name__ == "__main__":
    run_monitoring_pipeline()
"""

with open('monitor_drift.py', 'w', encoding='utf-8') as f:
    f.write(monitoring_code)

print("✅ monitor_drift.py updated for v0.4.33")
print(f"Size: {os.path.getsize('monitor_drift.py')} bytes")