import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import json
import hashlib
from pathlib import Path
import plotly.graph_objects as go

# ==============================
# PATHS & CONSTANTS (FIXED)
# ==============================
BASE_DIR = Path(__file__).resolve().parent   # ✅ FIXED
MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"
USERS_DB_PATH = BASE_DIR / "users.json"

EMOTION_MAPPING = {
    0: "Fear 😨",
    1: "Happy 😊",
    2: "Sad 😢",
}

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Brainwave Emotion Analysis",
    page_icon="🧠",
    layout="wide",
)

# ==============================
# LOAD MODEL (WITH SAFE FALLBACK)
# ==============================
class MockModel:
    def predict(self, data):
        return np.random.randint(0, 3, size=len(data))

    def predict_proba(self, data):
        proba = np.random.rand(len(data), 3)
        return proba / proba.sum(axis=1, keepdims=True)


@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model, False
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return MockModel(), True


MODEL, USING_MOCK = load_model()

# ==============================
# SESSION INITIALISATION
# ==============================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "df" not in st.session_state:
    st.session_state.df = None
if "history" not in st.session_state:
    st.session_state.history = []
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"
if "dashboard_intro_done" not in st.session_state:
    st.session_state.dashboard_intro_done = False

# ==============================
# AUTH HELPERS
# ==============================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def load_users() -> dict:
    if not USERS_DB_PATH.exists():
        return {}
    with open(USERS_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users: dict) -> None:
    with open(USERS_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4)


def signup(username: str, password: str):
    users = load_users()
    if username in users:
        return False, "Username already exists."
    users[username] = {
        "password": hash_password(password),
        "created_at": time.time(),
    }
    save_users(users)
    return True, "Account created successfully 🎉"


def login(username: str, password: str):
    users = load_users()
    if username not in users:
        return False, "User not found."
    if users[username]["password"] != hash_password(password):
        return False, "Incorrect password."
    return True, "Logged in successfully ✅"

# ==============================
# SIDEBAR
# ==============================
def sidebar_menu():
    with st.sidebar:
        st.markdown("### 🧠 BrainWave01")
        st.caption("Brainwave Emotion Analyst")

        if USING_MOCK:
            st.warning("Demo mode enabled. Mock predictions.", icon="⚠️")
        else:
            st.success("XGBoost model loaded", icon="✅")

        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "📁 Upload & Predict", "👤 Profile", "ℹ️ About"],
        )

        dark_on = st.toggle("Dark mode", value=st.session_state.theme_mode == "dark")
        st.session_state.theme_mode = "dark" if dark_on else "light"

        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()

    return page

# ==============================
# DASHBOARD
# ==============================
def page_dashboard():
    st.title("🏠 Dashboard")
    st.caption("Overview of your activity and predictions")

    total_preds = len(st.session_state.history)
    last_emotion = st.session_state.history[-1]["pred_label"] if total_preds else "—"

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", total_preds)
    col2.metric("Last Predicted Emotion", last_emotion)
    col3.metric("Files Analyzed", len({h["file_name"] for h in st.session_state.history}))

# ==============================
# UPLOAD & PREDICT
# ==============================
def page_upload_predict():
    st.title("📁 Upload & Predict")

    uploaded_file = st.file_uploader("Upload EEG CSV", type=["csv", "xlsx"])

    if uploaded_file is None:
        st.info("Upload a file to start")
        return

    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.dataframe(df.head())

    selected_row = df.iloc[[0]]

    if st.button("✨ Predict Emotion"):
        if hasattr(MODEL, "predict_proba"):
            probs = MODEL.predict_proba(selected_row)[0]
            pred_class = int(np.argmax(probs))
        else:
            pred_class = int(MODEL.predict(selected_row)[0])
            probs = None

        pred_label = EMOTION_MAPPING[pred_class]
        st.success(f"Predicted Emotion: **{pred_label}**")

        st.session_state.history.append(
            {
                "file_name": uploaded_file.name,
                "pred_label": pred_label,
                "timestamp": time.time(),
            }
        )

# ==============================
# PROFILE
# ==============================
def page_profile():
    st.title("👤 Profile")
    st.write(f"Username: **{st.session_state.username}**")
    st.write(f"Total predictions: **{len(st.session_state.history)}**")

# ==============================
# ABOUT
# ==============================
def page_about():
    st.title("ℹ️ About")
    st.markdown(
        """
        **EEG Brainwave Emotion Analysis**
        
        - Uses ML models (XGBoost)
        - Predicts emotional states from EEG features
        - Deployed using Streamlit
        
        Built to bridge **Neuroscience + AI**.
        """
    )

# ==============================
# ENTRYPOINT
# ==============================
def main():
    if not st.session_state.authenticated:
        st.title("🧠 Brainwave Emotion Analysis")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            ok, msg = login(user, pwd)
            if ok:
                st.session_state.authenticated = True
                st.session_state.username = user
                st.rerun()
            else:
                st.error(msg)
        return

    page = sidebar_menu()

    if page.startswith("🏠"):
        page_dashboard()
    elif page.startswith("📁"):
        page_upload_predict()
    elif page.startswith("👤"):
        page_profile()
    else:
        page_about()


if __name__ == "__main__":
    main()
