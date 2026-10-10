# Task 15 — End-to-End Deep Learning Production Deployment

A CIFAR-10 image classification application built around an existing Keras model, a Flask inference API, Streamlit user interface, Docker containers, Kubernetes manifests, API tests, and lightweight runtime metrics.

## Architecture

```text
User/browser
    |
    v
Streamlit UI :8501
    |
    | HTTP multipart upload
    v
Flask API :5000 ----> Keras model (cifar10_model.keras)
    |                       |
    +-- /health             +-- 10 CIFAR-10 classes
    +-- /predict
    +-- /metrics
```

## 1. Prerequisites

- Windows 10/11, PowerShell
- Python 3.11 (for local execution)
- Docker Desktop with Linux containers enabled
- Optional: Minikube and kubectl for Kubernetes
- The supplied `cifar10_model(1).keras` model

## 2. Put the model in the project

Copy/rename the uploaded model into this exact location:

`backend/model/cifar10_model.keras`

The `model` folder is intentionally kept out of this text bundle if the model was not copied automatically. The backend Dockerfile expects this file to exist.

## 3. Run locally without Docker

Open PowerShell in the project directory.

Terminal 1 — API:

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:MODEL_PATH = "$PWD\model\cifar10_model.keras"
python app.py
```

If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, or run `.venv\Scripts\python.exe` directly.

Terminal 2 — UI:

```powershell
cd frontend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:API_URL = "http://127.0.0.1:5000"
streamlit run app.py
```

Open `http://localhost:8501`. API endpoints: `http://localhost:5000/`, `/health`, `/metrics`.

## 4. Test the API

With the API running, use a real image file:

```powershell
curl.exe -X POST "http://127.0.0.1:5000/predict" -F "file=@C:\path\to\image.jpg"
```

Run automated smoke tests (install `requests` first if running outside the backend environment):

```powershell
pip install requests
python -m unittest discover -s tests -v
```

Expected outcomes: `/` returns service metadata; `/health` returns HTTP 200 when model is loaded; `/predict` returns predicted class, confidence, class probabilities, and inference time; `/metrics` returns request/error counters and recent latency statistics. Exact predictions and timings depend on the image and hardware.

## 5. Docker Compose deployment

Ensure `backend/model/cifar10_model.keras` exists. From the project root:

```powershell
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f backend
```

Open Streamlit at `http://localhost:8501`. Verify:

```powershell
curl.exe http://localhost:5000/health
curl.exe http://localhost:5000/metrics
```

Stop:

```powershell
docker compose down
```

TensorFlow images can be large and the first build may take several minutes.

## 6. Kubernetes deployment with Minikube

Start the cluster:

```powershell
minikube start
kubectl get nodes
```

Build the images:

```powershell
docker build -t task15-cifar10-backend:latest .\backend
docker build -t task15-cifar10-frontend:latest .\frontend
```

Make the images available to Minikube. Choose ONE method:

```powershell
minikube image load task15-cifar10-backend:latest
minikube image load task15-cifar10-frontend:latest
```

Apply manifests:

```powershell
kubectl apply -f .\k8s\namespace.yaml
kubectl apply -f .\k8s\backend-deployment.yaml
kubectl apply -f .\k8s\frontend-deployment.yaml
kubectl -n task15-cifar10 get pods,svc
kubectl -n task15-cifar10 rollout status deployment/cifar10-backend --timeout=10m
kubectl -n task15-cifar10 rollout status deployment/cifar10-frontend --timeout=5m
```

Open the UI:

```powershell
minikube service cifar10-frontend -n task15-cifar10
```

If it does not open automatically, use port-forwarding in a separate terminal:

```powershell
kubectl -n task15-cifar10 port-forward service/cifar10-frontend 8501:8501
```

Then visit `http://localhost:8501`.

Useful diagnostics:

```powershell
kubectl -n task15-cifar10 get pods -o wide
kubectl -n task15-cifar10 describe pod <POD_NAME>
kubectl -n task15-cifar10 logs deployment/cifar10-backend
kubectl -n task15-cifar10 logs deployment/cifar10-frontend
kubectl -n task15-cifar10 get events --sort-by=.metadata.creationTimestamp
```

Clean up only this Task 15 namespace:

```powershell
kubectl delete namespace task15-cifar10
```

## 7. Screenshots required for submission

Capture your own screenshots after actually running the steps. Recommended names:

1. `01_project_structure.png` — source files in VS Code.
2. `02_model_loaded.png` — backend log showing model input/output shape.
3. `03_api_health.png` — `/health` response.
4. `04_prediction.png` — successful `/predict` response in Postman or terminal.
5. `05_streamlit_ui.png` — uploaded image and prediction.
6. `06_docker_compose.png` — `docker compose ps` with both services running.
7. `07_kubernetes_pods.png` — `kubectl ... get pods,svc`.
8. `08_kubernetes_ui.png` — Streamlit UI served from Kubernetes.
9. `09_metrics.png` — `/metrics` output.

Do not include fabricated screenshots or state a cloud URL unless deployed and verified.

## 8. Model-specific notes

- This API assumes a 32×32 RGB input and ten output scores in CIFAR-10 class order.
- It defaults to input pixel values `[0,255]`. If the model was trained with inputs divided by 255, set `NORMALIZE_INPUT=1` in the backend environment and rebuild/redeploy.
- If the model uses a different preprocessing pipeline, adjust `preprocess_image()` to exactly match training.
- CIFAR-10 is a small-image dataset. Predictions on general phone photos may be weak; use test images similar to the training data.
- This project reports runtime inference latency and request/error counts. It does not claim a validated accuracy score unless evaluated on a labeled test set.

## 9. GitHub publishing

From the project root, after adding the model file:

```powershell
git init
git add .
git commit -m "Complete Task 15 end-to-end deep learning deployment"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
git push -u origin main
```

Do not commit secrets. Model files may exceed GitHub's regular 100 MB file limit; if so, use Git LFS or publish model instructions/download link instead. Ensure you have rights to publish the model.

## 10. Final LMS checklist

- [ ] Source code and `requirements.txt`
- [ ] Keras model artifact (or documented model retrieval method)
- [ ] Dockerfiles and `docker-compose.yml`
- [ ] Kubernetes YAML files
- [ ] README with reproducible commands
- [ ] API tests and captured outputs
- [ ] Screenshots from real runs
- [ ] Final PDF report
- [ ] Public GitHub URL tested in a private/incognito window
- [ ] Submit PDF and/or repository URL through L&T Edutech LMS
