import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import time
import os

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LeafGuard AI – Tomato Disease Detector",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Class Labels ───────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites (Two-Spotted Spider Mite)",
    "Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Healthy",
]

DISEASE_INFO = {
    "Bacterial Spot": {
        "icon": "🦠",
        "severity": "High",
        "color": "#FF6B6B",
        "description": "Caused by Xanthomonas bacteria. Produces small, water-soaked spots that turn dark brown.",
        "treatment": "Apply copper-based bactericides. Remove infected leaves and avoid overhead irrigation.",
    },
    "Early Blight": {
        "icon": "🍂",
        "severity": "Medium",
        "color": "#FFB347",
        "description": "Caused by Alternaria solani fungus. Creates concentric ring patterns on leaves.",
        "treatment": "Use fungicides containing chlorothalonil or mancozeb. Rotate crops annually.",
    },
    "Late Blight": {
        "icon": "💧",
        "severity": "Critical",
        "color": "#FF4040",
        "description": "Caused by Phytophthora infestans. Rapidly destroys foliage in wet conditions.",
        "treatment": "Apply copper fungicides immediately. Remove and destroy infected plants.",
    },
    "Leaf Mold": {
        "icon": "🌿",
        "severity": "Medium",
        "color": "#90BE6D",
        "description": "Caused by Passalora fulva fungus. Causes yellowing on leaf surfaces with gray-purple mold below.",
        "treatment": "Improve air circulation. Apply fungicides and reduce humidity in greenhouses.",
    },
    "Septoria Leaf Spot": {
        "icon": "🔴",
        "severity": "High",
        "color": "#FF6B6B",
        "description": "Caused by Septoria lycopersici fungus. Creates small circular spots with dark borders.",
        "treatment": "Apply fungicides and remove infected lower leaves. Avoid wetting foliage.",
    },
    "Spider Mites (Two-Spotted Spider Mite)": {
        "icon": "🕷️",
        "severity": "Medium",
        "color": "#FFA500",
        "description": "Tiny mites that cause stippling and yellowing of leaves under hot, dry conditions.",
        "treatment": "Spray with miticides or neem oil. Increase humidity and use predatory mites.",
    },
    "Target Spot": {
        "icon": "🎯",
        "severity": "Medium",
        "color": "#FFB347",
        "description": "Caused by Corynespora cassiicola. Produces bull's-eye like brown lesions.",
        "treatment": "Apply chlorothalonil fungicides. Practice crop rotation and remove debris.",
    },
    "Tomato Yellow Leaf Curl Virus": {
        "icon": "🌀",
        "severity": "Critical",
        "color": "#FF4040",
        "description": "Viral disease transmitted by whiteflies causing leaf curling and stunted growth.",
        "treatment": "Control whitefly populations. Remove infected plants. Use resistant varieties.",
    },
    "Healthy": {
        "icon": "✅",
        "severity": "None",
        "color": "#43AA8B",
        "description": "Your tomato plant appears healthy with no visible signs of disease.",
        "treatment": "Continue regular watering, fertilization, and monitoring for early signs of disease.",
    },
}

SEVERITY_BADGE = {
    "Critical": ("🔴", "#FF4040"),
    "High": ("🟠", "#FF6B6B"),
    "Medium": ("🟡", "#FFB347"),
    "None": ("🟢", "#43AA8B"),
}

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* === App Background === */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        min-height: 100vh;
    }

    /* === Hide default Streamlit branding === */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* === Hero Banner === */
    .hero-banner {
        background: linear-gradient(135deg, rgba(255,107,107,0.15) 0%, rgba(78,205,196,0.15) 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B6B, #FFE66D, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.65);
        margin-top: 0.5rem;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,107,107,0.2);
        border: 1px solid rgba(255,107,107,0.4);
        color: #FF6B6B;
        padding: 0.3rem 1rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 1rem;
    }

    /* === Cards === */
    .glass-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }

    /* === Upload Area === */
    .upload-section {
        background: linear-gradient(135deg, rgba(78,205,196,0.1) 0%, rgba(255,107,107,0.1) 100%);
        border: 2px dashed rgba(78,205,196,0.4);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }

    /* === Result Cards === */
    .result-card {
        border-radius: 16px;
        padding: 1.8rem;
        margin-top: 1rem;
        border: 1px solid rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
    }
    .result-disease {
        font-size: 1.6rem;
        font-weight: 700;
        color: #fff;
        margin: 0.3rem 0;
    }
    .result-confidence {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    /* === Probability Bars === */
    .prob-bar-container {
        margin: 0.4rem 0;
    }
    .prob-label {
        display: flex;
        justify-content: space-between;
        color: rgba(255,255,255,0.85);
        font-size: 0.82rem;
        margin-bottom: 0.2rem;
    }
    .prob-bar-bg {
        background: rgba(255,255,255,0.08);
        border-radius: 50px;
        height: 8px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 50px;
        background: linear-gradient(90deg, #4ECDC4, #FF6B6B);
        transition: width 0.6s ease;
    }

    /* === Info Pills === */
    .info-pill {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    /* === Sidebar === */
    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 1rem;
    }
    .sidebar-class-item {
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        margin: 0.2rem 0;
        background: rgba(255,255,255,0.05);
        font-size: 0.82rem;
        color: rgba(255,255,255,0.8);
    }

    /* === Step Labels === */
    .step-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #4ECDC4;
        margin-bottom: 0.5rem;
    }

    /* Override Streamlit file uploader default styling */
    .stFileUploader > div {
        background: transparent !important;
        border: none !important;
    }

    div[data-testid="stFileUploadDropzone"] {
        background: rgba(255,255,255,0.03) !important;
        border: 2px dashed rgba(78,205,196,0.35) !important;
        border-radius: 14px !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B6B, #FF8E53) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(255,107,107,0.4) !important;
    }

    /* Metric Override */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    [data-testid="metric-container"] label {
        color: rgba(255,255,255,0.6) !important;
        font-size: 0.82rem !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #fff !important;
        font-size: 1.4rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Model Loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "Model", "tomato_leaf_model.h5")
    model = tf.keras.models.load_model(model_path)
    return model


def preprocess_image(image: Image.Image, target_size=(224, 224)):
    img = image.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, image: Image.Image):
    processed = preprocess_image(image)
    preds = model.predict(processed, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    return preds, top_idx, CLASS_NAMES[top_idx]


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sidebar-header'>🍅 LeafGuard AI</div>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<div class='step-label'>📋 Detectable Diseases</div>", unsafe_allow_html=True)
    for cls in CLASS_NAMES:
        info = DISEASE_INFO.get(cls, {})
        icon = info.get("icon", "•")
        st.markdown(
            f"<div class='sidebar-class-item'>{icon}&nbsp;&nbsp;{cls}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("""
    <div style='color:rgba(255,255,255,0.5);font-size:0.78rem;line-height:1.6'>
    <b style='color:rgba(255,255,255,0.8)'>How to use:</b><br>
    1. Upload a clear photo of a tomato leaf<br>
    2. Click <b>Analyse Leaf</b><br>
    3. Review diagnosis & treatment advice<br><br>
    <b style='color:rgba(255,255,255,0.8)'>Tips:</b><br>
    • Ensure good lighting<br>
    • Crop tightly around the leaf<br>
    • Avoid blurry images
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='color:rgba(255,255,255,0.35);font-size:0.72rem;text-align:center'>"
        "Powered by TensorFlow & Streamlit<br>Model accuracy may vary.</div>",
        unsafe_allow_html=True,
    )


# ─── Hero Banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
    <p class='hero-title'>🍅 LeafGuard AI</p>
    <p class='hero-subtitle'>AI-powered Tomato Leaf Disease Detection in seconds</p>
    <span class='hero-badge'>🧠 Deep Learning • 9 Classes • Instant Results</span>
</div>
""", unsafe_allow_html=True)


# ─── Main Layout ────────────────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1.3], gap="large")

with col_upload:
    st.markdown("<div class='step-label'>Step 1 · Upload Your Leaf Image</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="Drop a tomato leaf image here",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported: JPG, JPEG, PNG, WEBP",
        label_visibility="collapsed",
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)

        st.markdown("<div class='step-label' style='margin-top:1rem'>Step 2 · Run Analysis</div>", unsafe_allow_html=True)
        analyse = st.button("🔬 Analyse Leaf", use_container_width=True)
    else:
        st.markdown("""
        <div style='text-align:center;padding:3rem 1rem;color:rgba(255,255,255,0.4)'>
            <div style='font-size:3.5rem'>🌿</div>
            <div style='font-size:0.9rem;margin-top:0.5rem'>Upload a tomato leaf image to get started</div>
        </div>
        """, unsafe_allow_html=True)
        analyse = False

with col_result:
    st.markdown("<div class='step-label'>Step 3 · Diagnosis & Treatment</div>", unsafe_allow_html=True)

    if uploaded_file and analyse:
        # ── Loading ────────────────────────────────────────────────────────
        with st.spinner(""):
            progress_placeholder = st.empty()
            progress_placeholder.markdown("""
            <div style='text-align:center;padding:2rem;color:rgba(255,255,255,0.7)'>
                <div style='font-size:2rem;animation:spin 1s linear infinite'>⚙️</div>
                <div style='margin-top:0.8rem;font-size:0.9rem'>Loading AI model & analysing leaf…</div>
            </div>
            """, unsafe_allow_html=True)

            model = load_model()
            preds, top_idx, disease_name = predict(model, image)
            confidence = float(preds[top_idx]) * 100
            info = DISEASE_INFO[disease_name]
            severity = info["severity"]
            sev_icon, sev_color = SEVERITY_BADGE[severity]

        progress_placeholder.empty()

        # ── Result Card ────────────────────────────────────────────────────
        card_bg = info["color"] + "22"
        st.markdown(f"""
        <div class='result-card' style='background:{card_bg};border-color:{info["color"]}44'>
            <div style='font-size:0.8rem;color:rgba(255,255,255,0.6);margin-bottom:0.3rem'>DIAGNOSIS RESULT</div>
            <div style='display:flex;align-items:center;gap:0.5rem'>
                <span style='font-size:2rem'>{info["icon"]}</span>
                <span class='result-disease'>{disease_name}</span>
            </div>
            <div style='margin-top:0.8rem;display:flex;gap:1rem;flex-wrap:wrap'>
                <span class='info-pill' style='background:{sev_color}22;color:{sev_color};border:1px solid {sev_color}55'>
                    {sev_icon} Severity: {severity}
                </span>
                <span class='info-pill' style='background:rgba(78,205,196,0.15);color:#4ECDC4;border:1px solid rgba(78,205,196,0.3)'>
                    🎯 Confidence: {confidence:.1f}%
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metrics ────────────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Confidence", f"{confidence:.1f}%")
        with m2:
            st.metric("Severity", severity)
        with m3:
            top3 = np.argsort(preds)[::-1][:3]
            st.metric("Top Predictions", f"{len(top3)} shown")

        # ── Description & Treatment ────────────────────────────────────────
        tab1, tab2, tab3 = st.tabs(["📖 About", "💊 Treatment", "📊 All Probabilities"])

        with tab1:
            st.markdown(f"""
            <div class='glass-card'>
                <div style='font-size:0.8rem;color:rgba(255,255,255,0.5);font-weight:600;margin-bottom:0.5rem'>ABOUT THIS CONDITION</div>
                <div style='color:rgba(255,255,255,0.85);line-height:1.7;font-size:0.92rem'>{info["description"]}</div>
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            st.markdown(f"""
            <div class='glass-card'>
                <div style='font-size:0.8rem;color:rgba(255,255,255,0.5);font-weight:600;margin-bottom:0.5rem'>RECOMMENDED TREATMENT</div>
                <div style='color:rgba(255,255,255,0.85);line-height:1.7;font-size:0.92rem'>{info["treatment"]}</div>
            </div>
            """, unsafe_allow_html=True)

        with tab3:
            sorted_indices = np.argsort(preds)[::-1]
            for i in sorted_indices:
                prob = float(preds[i]) * 100
                name = CLASS_NAMES[i]
                is_top = i == top_idx
                bar_color = info["color"] if is_top else "#4ECDC4"
                label_style = "font-weight:700;color:#fff" if is_top else "color:rgba(255,255,255,0.75)"
                st.markdown(f"""
                <div class='prob-bar-container'>
                    <div class='prob-label'>
                        <span style='{label_style}'>{DISEASE_INFO[name]['icon']} {name}</span>
                        <span style='color:rgba(255,255,255,0.6)'>{prob:.1f}%</span>
                    </div>
                    <div class='prob-bar-bg'>
                        <div class='prob-bar-fill' style='width:{prob:.1f}%;background:{bar_color}'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    elif not uploaded_file:
        st.markdown("""
        <div class='glass-card' style='text-align:center;padding:3rem 1rem'>
            <div style='font-size:3rem'>🔬</div>
            <div style='color:rgba(255,255,255,0.5);margin-top:0.8rem;font-size:0.9rem'>
                Upload a leaf image and click <b style='color:#4ECDC4'>Analyse Leaf</b><br>to see the AI diagnosis here.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='glass-card' style='text-align:center;padding:3rem 1rem'>
            <div style='font-size:3rem'>👆</div>
            <div style='color:rgba(255,255,255,0.5);margin-top:0.8rem;font-size:0.9rem'>
                Great! Now click <b style='color:#FF6B6B'>Analyse Leaf</b> on the left to run the AI diagnosis.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Bottom Info Strip ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
b1, b2, b3 = st.columns(3)
with b1:
    st.markdown("""
    <div class='glass-card' style='text-align:center'>
        <div style='font-size:1.8rem'>🧠</div>
        <div style='color:#fff;font-weight:700;font-size:0.9rem;margin-top:0.4rem'>Deep Learning</div>
        <div style='color:rgba(255,255,255,0.5);font-size:0.78rem'>CNN-based architecture</div>
    </div>
    """, unsafe_allow_html=True)
with b2:
    st.markdown("""
    <div class='glass-card' style='text-align:center'>
        <div style='font-size:1.8rem'>⚡</div>
        <div style='color:#fff;font-weight:700;font-size:0.9rem;margin-top:0.4rem'>Instant Analysis</div>
        <div style='color:rgba(255,255,255,0.5);font-size:0.78rem'>Results in under a second</div>
    </div>
    """, unsafe_allow_html=True)
with b3:
    st.markdown("""
    <div class='glass-card' style='text-align:center'>
        <div style='font-size:1.8rem'>🌿</div>
        <div style='color:#fff;font-weight:700;font-size:0.9rem;margin-top:0.4rem'>9 Disease Classes</div>
        <div style='color:rgba(255,255,255,0.5);font-size:0.78rem'>Including healthy detection</div>
    </div>
    """, unsafe_allow_html=True)
