import os, requests, streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:5000")
st.set_page_config(page_title="CIFAR-10 AI Classifier", page_icon="🧠", layout="centered")
st.title("🧠 CIFAR-10 Image Classification")
st.write("Upload an image. The Streamlit UI sends it to the Flask API for inference.")
st.caption("The model is trained for 32×32 CIFAR-10-style images. Results on unrelated/high-resolution photos may be unreliable.")

uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"])
if st.button("Check API health"):
    try:
        r = requests.get(f"{API_URL}/health", timeout=10)
        st.write(r.status_code, r.json())
    except requests.RequestException as e:
        st.error(f"API unavailable: {e}")

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Predict image", type="primary"):
        try:
            with st.spinner("Running prediction..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
                    timeout=120
                )
            if response.ok:
                result = response.json()
                st.success(f"Prediction: {result['predicted_class']}")
                st.metric("Confidence", f"{result['confidence']:.1%}")
                st.metric("Inference time", f"{result['inference_time_ms']:.2f} ms")
                st.subheader("Class probabilities")
                st.bar_chart(result["probabilities"])
                with st.expander("Raw API response"):
                    st.json(result)
            else:
                st.error(f"API returned {response.status_code}: {response.text}")
        except requests.RequestException as e:
            st.error(f"Could not contact Flask API: {e}")

st.divider()
st.subheader("Service monitoring")
if st.button("Load API metrics"):
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=10)
        st.json(r.json())
    except requests.RequestException as e:
        st.error(f"Could not load metrics: {e}")
