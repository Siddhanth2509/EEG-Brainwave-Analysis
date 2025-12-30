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
# PATHS & CONSTANTS
# ==============================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"
USERS_DB_PATH = Path(__file__).resolve().parent / "users.json"

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
# THEME & GLOBAL STYLES
# ==============================
def get_app_css(theme_mode: str) -> str:
    """
    Return CSS string for current theme.
    theme_mode: "dark" or "light"
    """
    if theme_mode == "light":
        accent = "#2563eb"
        bg_main = "#f3f4f6"
        text_color = "#111827"
        sidebar_bg = "#ffffff"
        card_bg1 = "#ffffff"
        card_bg2 = "#e5e7eb"
        border_color = "#d1d5db"
    else:  # dark default
        accent = "#60a5fa"
        bg_main = "#05070b"
        text_color = "#e5e7eb"
        sidebar_bg = "#05070c"
        card_bg1 = "#050814"
        card_bg2 = "#020308"
        border_color = "#151c27"

    css = """
<style>

/* ---------- GLOBAL CONTAINER ---------- */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, __BG_MAIN__, #020308);
    color: __TEXT_COLOR__ !important;
    font-family: "Poppins", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    position: relative;
    overflow-y: auto !important;
    height: 100vh !important;
    scroll-behavior: smooth;
}

/* remove default chrome */
header[data-testid="stHeader"] { display: none; }
footer[data-testid="stFooter"] { display: none; }
[data-testid="stToolbar"] { display: none; }

/* ---------- FLOATING NEURAL GLOWS ---------- */
[data-testid="stAppViewContainer"]::before,
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    border-radius: 999px;
    filter: blur(40px);
    opacity: 0.22;
    pointer-events: none;
    z-index: -1;
}
[data-testid="stAppViewContainer"]::before {
    width: 320px;
    height: 320px;
    top: -60px;
    right: -80px;
    background: radial-gradient(circle, __ACCENT__ 0%, transparent 70%);
    animation: floatOrb 18s infinite alternate ease-in-out;
}
[data-testid="stAppViewContainer"]::after {
    width: 260px;
    height: 260px;
    bottom: -60px;
    left: -40px;
    background: radial-gradient(circle, #0ea5e9 0%, transparent 70%);
    animation: floatOrb2 20s infinite alternate ease-in-out;
}

@keyframes floatOrb {
    0% { transform: translate3d(0,0,0); }
    100% { transform: translate3d(-30px,40px,0); }
}
@keyframes floatOrb2 {
    0% { transform: translate3d(0,0,0); }
    100% { transform: translate3d(50px,-30px,0); }
}

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"] {
    background: __SIDEBAR_BG__;
    border-right: 1px solid __BORDER_COLOR__;
}
[data-testid="stSidebar"] * {
    color: __TEXT_COLOR__ !important;
}

.sidebar-avatar {
    width: 72px;
    height: 72px;
    border-radius: 999px;
    background: radial-gradient(circle, __ACCENT__, #0f172a);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 0.8rem;
    box-shadow: 0 0 18px rgba(15, 118, 255, 0.35);
    border: 1px solid rgba(148, 163, 184, 0.5);
}
.sidebar-avatar span {
    font-size: 2rem;
}

/* ---------- CARDS & TITLES ---------- */
.metric-card {
    padding: 1.3rem 1.6rem;
    border-radius: 1rem;
    background: linear-gradient(135deg, __CARD_BG1__, __CARD_BG2__);
    border: 1px solid __BORDER_COLOR__;
    box-shadow: 0 3px 10px rgba(0,0,0,0.35);
    transition: 0.2s ease-in-out;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 22px rgba(0,0,0,0.6);
    border-color: __ACCENT__;
}

.app-title {
    font-size: 2.0rem;
    font-weight: 600;
    margin-bottom: 0.2rem;
    color: __ACCENT__;
}
.app-subtitle {
    font-size: 0.95rem;
    color: #94a1b2;
    margin-bottom: 1.1rem;
}

/* ---------- AUTH CARD ---------- */
.auth-card {
    max-width: 420px;
    margin: 4rem auto;
    padding: 2.2rem 2.0rem 2.4rem 2.0rem;
    border-radius: 1.4rem;
    background: radial-gradient(circle at top left, __CARD_BG1__, __CARD_BG2__);
    border: 1px solid __BORDER_COLOR__;
    box-shadow: 0 16px 40px rgba(0,0,0,0.8);
}

/* ---------- BUTTONS ---------- */
.stButton>button {
    border-radius: 0.6rem;
    padding: 0.55rem 1.3rem;
    background: #111827;
    border: 1px solid #1f2937;
    color: #cfd3d6;
    transition: 0.25s ease-in-out;
    font-weight: 500;
}
.stButton>button:hover {
    background: __ACCENT__;
    color: #0b1020;
    box-shadow: 0px 0px 10px __ACCENT__;
    transform: translateY(-1px) scale(1.02);
}

/* ---------- INPUTS / SELECTS ---------- */
.stTextInput>div>div>input,
.stSelectbox>div>div>div>div,
.stFileUploader>label {
    background: #020617 !important;
    border-radius: 0.55rem;
    border: 1px solid #1f2937 !important;
    color: #cfd3d6 !important;
}
.stFileUploader>label:hover {
    border-color: __ACCENT__ !important;
}

/* ---------- DATAFRAME ---------- */
[data-testid="stDataFrameContainer"] {
    background: rgba(15,23,42,0.7) !important;
    border-radius: 0.6rem;
    border: 1px solid __BORDER_COLOR__;
    padding: 6px;
}

/* ---------- LOADING WAVE ---------- */
.loading-wave {
    display: flex;
    gap: 4px;
    margin-top: 0.6rem;
}
.loading-bar {
    width: 6px;
    height: 18px;
    border-radius: 999px;
    background: __ACCENT__;
    animation: wave 1s infinite ease-in-out;
}
.loading-bar:nth-child(2) { animation-delay: 0.1s; }
.loading-bar:nth-child(3) { animation-delay: 0.2s; }
.loading-bar:nth-child(4) { animation-delay: 0.3s; }
.loading-bar:nth-child(5) { animation-delay: 0.4s; }

@keyframes wave {
    0%, 100% { transform: scaleY(0.4); opacity: 0.5; }
    50% { transform: scaleY(1.3); opacity: 1; }
}

/* ---------- NEURAL WAVE LINE ---------- */
.neural-wave {
    position: relative;
    height: 60px;
    margin-top: 1.4rem;
    overflow: hidden;
}
.neural-wave::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image: linear-gradient(90deg, transparent 0%, __ACCENT__ 40%, transparent 80%);
    background-size: 200% 100%;
    mix-blend-mode: screen;
    opacity: 0.35;
    animation: moveWave 3.4s infinite linear;
}
@keyframes moveWave {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* ---------- BRAIN DISKS ---------- */
.profile-brain, .page-brain {
    width: 180px;
    height: 180px;
    border-radius: 999px;
    background: radial-gradient(circle at 30% 30%, __ACCENT__, transparent 60%), 
                radial-gradient(circle at 70% 70%, rgba(56,189,248,0.5), transparent 65%),
                #020617;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3.5rem;
    margin: 10px auto;
    box-shadow: 0 0 15px rgba(15,23,42,0.9);
    position: relative;
    overflow: hidden;
    transition: 0.35s ease-in-out;
}
.profile-brain::after, .page-brain::after {
    content: "";
    position: absolute;
    width: 220%;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(148,163,184,0.7), transparent);
    top: 50%;
    left: -60%;
    opacity: 0.35;
    transform: translateY(-50%);
    animation: neuronFlow 4s infinite linear;
}
@keyframes neuronFlow {
    0% { transform: translate(-60%, -50%); }
    100% { transform: translate(60%, -50%); }
}
.profile-brain:hover, .page-brain:hover {
    box-shadow: 0 0 28px rgba(96,165,250,0.6);
    transform: scale(1.03);
}

/* ---------- CURSORS ---------- */
* { cursor: default; }
.stButton>button,
.stSelectbox,
.stFileUploader>label,
.stTextInput>div>div>input {
    cursor: pointer;
}
</style>
"""
    css = (
        css.replace("__ACCENT__", accent)
        .replace("__BG_MAIN__", bg_main)
        .replace("__TEXT_COLOR__", text_color)
        .replace("__SIDEBAR_BG__", sidebar_bg)
        .replace("__CARD_BG1__", card_bg1)
        .replace("__CARD_BG2__", card_bg2)
        .replace("__BORDER_COLOR__", border_color)
    )
    return css


# ==============================
# SIMPLE AUTH SYSTEM
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
        return False, "Username already exists. Please choose another."
    users[username] = {
        "password": hash_password(password),
        "created_at": time.time(),
    }
    save_users(users)
    return True, "Account created successfully! 🎉"


def login(username: str, password: str):
    users = load_users()
    if username not in users:
        return False, "User not found. Please sign up."
    if users[username]["password"] != hash_password(password):
        return False, "Incorrect password. Try again."
    return True, "Logged in successfully ✅"


# ==============================
# LOAD MODEL (WITH FALLBACK)
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
    except Exception:
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
    st.session_state.theme_mode = "dark"  # dark by default
if "dashboard_intro_done" not in st.session_state:
    st.session_state.dashboard_intro_done = False

# Inject CSS according to theme
st.markdown(get_app_css(st.session_state.theme_mode), unsafe_allow_html=True)


# ==============================
# SMALL HELPERS
# ==============================
def typewriter(text: str, key: str, speed: float = 0.03):
    """Simple typewriter effect; runs once per session for given key."""
    placeholder = st.empty()
    flag_key = f"{key}_done"
    if st.session_state.get(flag_key):
        placeholder.markdown(text)
        return

    out = ""
    for ch in text:
        out += ch
        placeholder.markdown(out)
        time.sleep(speed)
    st.session_state[flag_key] = True


def apply_plotly_theme(fig: go.Figure):
    """Make Plotly charts match dark/light theme."""
    if st.session_state.theme_mode == "dark":
        font_color = "#e5e7eb"
    else:
        font_color = "#111827"

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=font_color),
    )


# ==============================
# AUTH SCREEN
# ==============================
def auth_screen():
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.markdown('<div class="app-title">🧠 Brainwave Emotion Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Log in or create an account to start predicting emotions from EEG data.</div>',
        unsafe_allow_html=True,
    )

    tab_login, tab_signup = st.tabs(["🔑 Login", "🆕 Sign Up"])

    with tab_login:
        login_user = st.text_input("Username", key="login_username")
        login_pass = st.text_input("Password", type="password", key="login_password")
        if st.button("Log In"):
            ok, msg = login(login_user, login_pass)
            if ok:
                st.success(msg)
                st.session_state.authenticated = True
                st.session_state.username = login_user
                st.rerun()
            else:
                st.error(msg)

    with tab_signup:
        new_user = st.text_input("Choose a username", key="signup_username")
        new_pass = st.text_input("Choose a password", type="password", key="signup_password")
        new_pass2 = st.text_input("Confirm password", type="password", key="signup_password2")

        if st.button("Create account"):
            if not new_user or not new_pass:
                st.warning("Username and password can’t be empty.")
            elif new_pass != new_pass2:
                st.error("Passwords do not match.")
            else:
                ok, msg = signup(new_user, new_pass)
                if ok:
                    st.success(msg)
                    st.session_state.authenticated = True
                    st.session_state.username = new_user
                    st.rerun()
                else:
                    st.error(msg)

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================
# MAIN APP PAGES
# ==============================
def sidebar_menu():
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-avatar">
                <span>🧑‍💻</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"**{st.session_state.username}**")
        st.caption("Brainwave Emotion Analyst")

        if USING_MOCK:
            st.warning("Using mock model (xgboost_model.pkl not found). Predictions are random.", icon="⚠️")

        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "📁 Upload & Predict", "👤 Profile", "ℹ️ About"],
            index=0,
        )

        st.markdown("---")

        # Day/Night toggle
        dark_default = st.session_state.theme_mode == "dark"
        dark_on = st.toggle("Dark mode", value=dark_default)
        new_mode = "dark" if dark_on else "light"
        if new_mode != st.session_state.theme_mode:
            st.session_state.theme_mode = new_mode
            st.rerun()

        # Collapsible "neural map"
        with st.expander("Neural map"):
            st.caption(
                "A conceptual view of how EEG features flow through the model: "
                "brain signals → frequency bands → ML model → emotion probabilities."
            )

        st.markdown("---")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
    return page


def page_dashboard():
    st.markdown('<div class="app-title">🏠 Dashboard</div>', unsafe_allow_html=True)
    typewriter("Welcome to your neural emotion analysis dashboard.", "dashboard_intro")
    st.markdown('<div class="app-subtitle">Overview of your activity and model insights.</div>', unsafe_allow_html=True)

    # Subtle brain visual
    st.markdown('<div class="page-brain">🧠</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    total_preds = len(st.session_state.history)
    last_emotion = st.session_state.history[-1]["pred_label"] if total_preds > 0 else "—"
    uniq_files = len({h["file_name"] for h in st.session_state.history}) if total_preds > 0 else 0

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Predictions", total_preds)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Last Predicted Emotion", last_emotion)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Files Analyzed", uniq_files)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="neural-wave"></div>', unsafe_allow_html=True)

    st.markdown("### Recent Predictions")
    if total_preds == 0:
        st.info("No predictions yet. Go to **Upload & Predict** to start.")
        return

    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df.tail(10), use_container_width=True)

    # 3D-style emotion distribution
    st.markdown("### Emotion Distribution (3D View)")
    counts = hist_df["pred_label"].value_counts()
    emotions = list(counts.index)
    x = list(range(len(emotions)))
    y = [0] * len(emotions)
    z = list(counts.values)

    fig3d = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers+lines",
                marker=dict(size=10, color=z, colorscale="Blues"),
                line=dict(width=14),
            )
        ]
    )
    fig3d.update_layout(
        scene=dict(
            xaxis=dict(
                title="Emotion",
                tickmode="array",
                tickvals=x,
                ticktext=emotions,
            ),
            yaxis=dict(title=""),
            zaxis=dict(title="Count"),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
    )
    apply_plotly_theme(fig3d)
    st.plotly_chart(fig3d, use_container_width=True)

    # Timeline chart
    st.markdown("### Prediction Timeline")
    hist_df["timestamp_dt"] = pd.to_datetime(hist_df["timestamp"])
    hist_df_sorted = hist_df.sort_values("timestamp_dt")
    fig_tl = go.Figure(
        data=[
            go.Scatter(
                x=hist_df_sorted["timestamp_dt"],
                y=list(range(len(hist_df_sorted))),
                mode="lines+markers",
                text=hist_df_sorted["pred_label"],
                hovertemplate="Time: %{x}<br>Emotion: %{text}<extra></extra>",
            )
        ]
    )
    fig_tl.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Prediction index",
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    apply_plotly_theme(fig_tl)
    st.plotly_chart(fig_tl, use_container_width=True)


def page_upload_predict():
    st.markdown('<div class="app-title">📁 Upload & Predict</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Upload EEG feature data, select a record, and predict the emotional state.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx"],
        help="File should contain EEG features. Optionally include a 'subject_id' column.",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df = df
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            st.session_state.df = None

    df = st.session_state.df
    if df is None:
        st.info("Upload a file to begin.")
        return

    st.success("File uploaded successfully ✅")

    with st.expander("Preview data"):
        st.dataframe(df.head(), use_container_width=True)

    st.markdown("### Select a record")

    if "subject_id" in df.columns:
        options = df["subject_id"].astype(str).tolist()
        selected = st.selectbox("Choose subject_id", options)
        selected_row = df[df["subject_id"].astype(str) == selected]
    else:
        options = df.index.tolist()
        selected = st.selectbox("Choose row index", options)
        selected_row = df.loc[[selected]]

    st.write("#### Selected Record")
    st.dataframe(selected_row, use_container_width=True)

    st.markdown("### Predict Emotion")

    if st.button("✨ Predict Emotion"):
        with st.spinner("Analyzing brainwave patterns..."):
            st.markdown(
                """
                <div class="loading-wave">
                    <div class="loading-bar"></div>
                    <div class="loading-bar"></div>
                    <div class="loading-bar"></div>
                    <div class="loading-bar"></div>
                    <div class="loading-bar"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            model_input = selected_row.copy()
            if "subject_id" in model_input.columns:
                model_input = model_input.drop(columns=["subject_id"])

            time.sleep(1.4)  # visual delay

            try:
                if hasattr(MODEL, "predict_proba"):
                    probs = MODEL.predict_proba(model_input)[0]
                    pred_class = int(np.argmax(probs))
                else:
                    pred_class = int(MODEL.predict(model_input)[0])
                    probs = None

                pred_label = EMOTION_MAPPING.get(pred_class, "Unknown 🤔")

                col1, col2 = st.columns([2, 3])
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Predicted Emotional State", pred_label)
                    st.markdown("</div>", unsafe_allow_html=True)

                if probs is not None:
                    prob_df = pd.DataFrame(
                        {"Emotion": list(EMOTION_MAPPING.values()), "Probability": probs}
                    ).set_index("Emotion")
                    with col2:
                        st.bar_chart(prob_df)

                # Save to history
                st.session_state.history.append(
                    {
                        "user": st.session_state.username,
                        "file_name": uploaded_file.name,
                        "record_id": str(selected),
                        "pred_label": pred_label,
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")


def page_profile():
    st.markdown('<div class="app-title">👤 Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Your neural identity panel.</div>', unsafe_allow_html=True)

    st.markdown('<div class="profile-brain">🧠</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("### Account")

    col1, col2 = st.columns(2)
    users = load_users()
    created_ts = users.get(st.session_state.username, {}).get("created_at", None)

    with col1:
        st.write(f"**Username:** {st.session_state.username}")
    with col2:
        if created_ts:
            created = pd.to_datetime(created_ts, unit="s")
            st.write(f"**Joined:** {created.strftime('%d %B %Y')}")

    st.markdown("### Prediction Stats")
    st.write(f"**Total predictions:** {len(st.session_state.history)}")


def page_about():
    st.markdown('<div class="app-title">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown(
        """
        This application demonstrates **emotion prediction from EEG brainwave data**
        using machine learning models.

        **Pipeline:**
        - EEG feature extraction from frequency bands (e.g., via FFT)
        - Ensemble models: XGBoost, LightGBM, Random Forest, AdaBoost
        - Deployed with a modern Streamlit dashboard for interaction

        The goal is to bridge **Neuroscience** and **Artificial Intelligence** to better
        understand and visualize emotional states from brain activity.
        """,
        unsafe_allow_html=False,
    )
    st.markdown('<div class="page-brain">🧠</div>', unsafe_allow_html=True)


# ==============================
# APP ENTRYPOINT
# ==============================
def main():
    if not st.session_state.authenticated:
        auth_screen()
        return

    page = sidebar_menu()

    if page.startswith("🏠"):
        page_dashboard()
    elif page.startswith("📁"):
        page_upload_predict()
    elif page.startswith("👤"):
        page_profile()
    elif page.startswith("ℹ️"):
        page_about()


if __name__ == "__main__":
    main()
