import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Configuration ---
MODEL_PATH = "waste_classifier_v1.keras"
IMG_SIZE = (224, 224)
PREPROCESS_MODE = "raw"

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Color-coded metadata for each waste class
WASTE_INFO = {
    "cardboard": {
        "emoji": "📦",
        "color": "#D97706",
        "bg": "#FEF3C7",
        "bin": "Blue Recycling Bin",
        "recyclable": True,
        "tip": "Flatten boxes to save space and keep them dry and grease-free.",
    },
    "glass": {
        "emoji": "🍾",
        "color": "#059669",
        "bg": "#D1FAE5",
        "bin": "Green / Blue Recycling Bin",
        "recyclable": True,
        "tip": "Rinse bottles and jars thoroughly; remove metal or plastic lids.",
    },
    "metal": {
        "emoji": "🥫",
        "color": "#2563EB",
        "bg": "#DBEAFE",
        "bin": "Blue Recycling Bin",
        "recyclable": True,
        "tip": "Empty and rinse cans; crush aluminum cans if possible.",
    },
    "paper": {
        "emoji": "📄",
        "color": "#0284C7",
        "bg": "#E0F2FE",
        "bin": "Blue Recycling Bin",
        "recyclable": True,
        "tip": "Keep paper clean and dry; avoid recycling food-soiled or waxed paper.",
    },
    "plastic": {
        "emoji": "🧴",
        "color": "#7C3AED",
        "bg": "#EDE9FE",
        "bin": "Blue Recycling Bin",
        "recyclable": True,
        "tip": "Empty, rinse, and check local resin codes (#1, #2, #5 are widely recycled).",
    },
    "trash": {
        "emoji": "🗑️",
        "color": "#DC2626",
        "bg": "#FEE2E2",
        "bin": "Black General Waste Bin",
        "recyclable": False,
        "tip": "Non-recyclable item—dispose of responsibly in general waste.",
    },
}

# --- Page Setup ---
st.set_page_config(
    page_title="EcoSort AI | Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
)

# --- Custom CSS Styling ---
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #F0FDF4 0%, #EFF6FF 50%, #FAF5FF 100%);
    }
    .hero-banner {
        background: linear-gradient(90deg, #059669 0%, #0D9488 50%, #2563EB 100%);
        padding: 1.8rem 2rem;
        border-radius: 18px;
        color: white;
        box-shadow: 0 10px 25px rgba(5, 150, 105, 0.2);
        margin-bottom: 1.5rem;
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        color: white;
    }
    .hero-banner p {
        margin: 0.4rem 0 0 0;
        font-size: 1.05rem;
        opacity: 0.92;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 16px;
        border-left: 8px solid;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
    }
    .pill-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.6rem;
    }
    .prob-row {
        margin-bottom: 0.75rem;
    }
    .prob-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.92rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
        color: #1F2937;
    }
    .prob-track {
        width: 100%;
        height: 12px;
        background-color: #E5E7EB;
        border-radius: 999px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.4s ease;
    }
    .category-chip {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 10px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session State for Live Demo Stats ---
if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0
if "recyclable_count" not in st.session_state:
    st.session_state.recyclable_count = 0
if "last_image_id" not in st.session_state:
    st.session_state.last_image_id = None

# --- Model Loader ---
@st.cache_resource
def load_waste_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    if PREPROCESS_MODE == "rescale":
        return arr / 255.0
    elif PREPROCESS_MODE == "mobilenet_v2":
        return tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    elif PREPROCESS_MODE == "resnet50":
        return tf.keras.applications.resnet50.preprocess_input(arr)
    elif PREPROCESS_MODE == "efficientnet":
        return tf.keras.applications.efficientnet.preprocess_input(arr)
    return arr  # "raw" (uses built-in Rescaling layer)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🌱 EcoSort AI")
    st.caption("Transfer Learning Waste Classifier (MobileNetV2)")
    st.divider()

    st.markdown("### 📊 Session Impact")
    s1, s2 = st.columns(2)
    s1.metric("Items Scanned", st.session_state.scan_count)
    s2.metric("Recyclable", st.session_state.recyclable_count)

    if st.button("Reset Counter", use_container_width=True):
        st.session_state.scan_count = 0
        st.session_state.recyclable_count = 0
        st.session_state.last_image_id = None
        st.rerun()

    st.divider()
    st.markdown("### 🎨 Waste Categories")
    chips_html = ""
    for name in CLASS_NAMES:
        info = WASTE_INFO[name]
        chips_html += (
            f"<span class='category-chip' style='background:{info['bg']}; color:{info['color']}; "
            f"border: 1px solid {info['color']}40;'>{info['emoji']} {name.capitalize()}</span>"
        )
    st.markdown(chips_html, unsafe_allow_html=True)

    st.divider()
    st.markdown("### ⚙️ Model Specs")
    st.markdown(
        "- **Backbone:** MobileNetV2 (224×224)\n"
        "- **Classes:** 6 Waste Types\n"
        "- **Preprocessing:** Built-in Rescaling Layer"
    )

# --- Hero Header ---
st.markdown(
    """
    <div class="hero-banner">
        <h1>♻️ EcoSort AI — Smart Waste Classifier</h1>
        <p>Upload a photo or use your live camera to identify waste and get instant recycling guidance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Load Model ---
try:
    model = load_waste_model()
except Exception as e:
    st.error(f"Could not load `{MODEL_PATH}`. Make sure it is in the same folder as `app.py`.\n\nError: {e}")
    st.stop()

# --- Input Section ---
tab_upload, tab_cam = st.tabs(["📁 Upload Waste Photo", "📷 Live Camera Scan"])

uploaded_image = None
current_id = None

with tab_upload:
    file = st.file_uploader(
        "Drag and drop or browse a waste image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
    )
    if file:
        uploaded_image = Image.open(file)
        current_id = f"upload_{file.name}_{file.size}"

with tab_cam:
    cam_file = st.camera_input("Point your camera at a waste item and capture")
    if cam_file:
        uploaded_image = Image.open(cam_file)
        current_id = f"cam_{cam_file.size}"

# --- Prediction & Colorful Results ---
if uploaded_image is not None:
    with st.spinner("Analyzing waste item..."):
        input_tensor = preprocess_image(uploaded_image)
        preds = model.predict(input_tensor, verbose=0)[0]

        if not np.isclose(np.sum(preds), 1.0, atol=1e-2):
            preds = tf.nn.softmax(preds).numpy()

        top_idx = int(np.argmax(preds))
        label = CLASS_NAMES[top_idx]
        confidence = float(preds[top_idx]) * 100
        info = WASTE_INFO[label]

        # Update session stats only once per unique image
        if current_id != st.session_state.last_image_id:
            st.session_state.scan_count += 1
            if info["recyclable"]:
                st.session_state.recyclable_count += 1
            st.session_state.last_image_id = current_id

    st.markdown("---")
    col_img, col_res = st.columns([1, 1.3], gap="large")

    with col_img:
        st.markdown("#### 🖼️ Scanned Item")
        st.image(uploaded_image, use_container_width=True)

    with col_res:
        badge_text = "♻️ RECYCLABLE" if info["recyclable"] else "🗑️ GENERAL WASTE"
        st.markdown(
            f"""
            <div class="result-card" style="background-color: {info['bg']}; border-left-color: {info['color']};">
                <span class="pill-badge" style="background-color: {info['color']};">{badge_text}</span>
                <h2 style="margin: 0.2rem 0; color: #111827;">{info['emoji']} {label.upper()}</h2>
                <p style="font-size: 1.15rem; font-weight: 700; color: {info['color']}; margin: 0.2rem 0 0.8rem 0;">
                    Confidence: {confidence:.1f}%
                </p>
                <p style="margin: 0; color: #374151; font-size: 0.98rem;">
                    <strong>Recommended Bin:</strong> {info['bin']}<br/>
                    <strong>Disposal Tip:</strong> {info['tip']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📈 Class Confidence Breakdown")
        sorted_indices = np.argsort(preds)[::-1]
        for idx in sorted_indices[:3]:
            cname = CLASS_NAMES[idx]
            cinfo = WASTE_INFO[cname]
            pct = float(preds[idx]) * 100
            st.markdown(
                f"""
                <div class="prob-row">
                    <div class="prob-label">
                        <span>{cinfo['emoji']} {cname.capitalize()}</span>
                        <span>{pct:.1f}%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill" style="width: {pct:.1f}%; background-color: {cinfo['color']};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("View all 6 class probabilities"):
            for idx in sorted_indices:
                cname = CLASS_NAMES[idx]
                cinfo = WASTE_INFO[cname]
                pct = float(preds[idx]) * 100
                st.write(f"{cinfo['emoji']} **{cname.capitalize()}**: `{pct:.2f}%`")
else:
    st.info("👆 Upload an image or switch to the **Live Camera Scan** tab to classify a waste item.")