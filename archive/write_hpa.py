import os

# Ensure directory is present
os.makedirs('k8s', exist_ok=True)

hpa_content = """apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-autoscaler
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  minReplicas: 2
  maxReplicas: 6
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 50
"""

with open('k8s/hpa.yaml', 'w') as f:
    f.write(hpa_content)

print("✅ k8s/hpa.yaml auto-scaler configuration written.")
print(f"Size: {os.path.getsize('k8s/hpa.yaml')} bytes")