import os

monitoring_code = """import os
import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metric_preset import TextOverviewPreset
from evidently.metrics import TextDataDriftMetric

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
        
        # 3. Simulate Production Data (Real incoming traffic - displaying textual shift)
        production_data = pd.DataFrame({
            "user_query": [
                "Who is playing the football match in Riyadh tonight?",
                "Can you recommend a cheap hotel near the airport?",
                "What is the population size of the kingdom right now?",
                "How to apply for an international tourist e-visa?",
                "Show me the stock pricing charts for oil companies."
            ]
        })
        
        print("Calculating Text Data Drift Metrics via Evidently AI...")
        
        # 4. Initialize and compile the Text Metrics Report
        text_report = Report(metrics=[
            TextDataDriftMetric(column_name="user_query")
        ])
        text_report.run(reference_data=reference_data, current_data=production_data)
        
        # 5. Extract calculated parameters out of the report dictionary safely
        report_dict = text_report.as_dict()
        drift_share = report_dict["metrics"][0]["result"]["drift_share"]
        drift_detected = report_dict["metrics"][0]["result"]["dataset_drift"]
        
        # 6. Log metrics directly onto your active tracking backend
        mlflow.log_metric("drift_share", drift_share)
        mlflow.log_param("drift_detected", str(drift_detected))
        
        print(f"📊 Tracking complete! Drift Share: {drift_share:.4f} | Drift Detected: {drift_detected}")
        
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

print("✅ monitor_drift.py written down cleanly.")
print(f"Size: {os.path.getsize('monitor_drift.py')} bytes")