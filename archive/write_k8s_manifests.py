import os

# Ensure the directory exists
os.makedirs('k8s', exist_ok=True)

# 1. Write configmap.yaml
configmap_content = """apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
data:
  qdrant_url: "http://rag-qdrant-service:6333"
"""

with open('k8s/configmap.yaml', 'w') as f:
    f.write(configmap_content)
print("✅ k8s/configmap.yaml written.")

# 2. Write service.yaml
service_content = """apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
spec:
  type: NodePort
  selector:
    app: rag-api
  ports:
    - port: 80
      targetPort: 80
      nodePort: 30080
"""

with open('k8s/service.yaml', 'w') as f:
    f.write(service_content)
print("✅ k8s/service.yaml written.")

# 3. Write deployment.yaml
deployment_content = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  labels:
    app: rag-api
    version: "1.0.0"
spec:
  replicas: 2
  selector:
    matchLabels:
      app: rag-api
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: rag-api
        version: "1.0.0"
    spec:
      containers:
      - name: rag-api
        image: nginx:alpine
        ports:
        - containerPort: 80
        env:
        - name: QDRANT_URL
          valueFrom:
            configMapKeyRef:
              name: rag-config
              key: qdrant_url
        resources:
          requests:
            memory: "64Mi"
            cpu: "100m"
          limits:
            memory: "128Mi"
            cpu: "200m"
        livenessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 10
          periodSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 5
          failureThreshold: 3
"""

with open('k8s/deployment.yaml', 'w') as f:
    f.write(deployment_content)
print("✅ k8s/deployment.yaml written.")
print("\\nAll manifests synchronized completely.")