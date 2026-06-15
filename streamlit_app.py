import uuid
import requests
import pandas as pd
import streamlit as st


API_URL_DEFAULT = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Agent MCP CRM",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1350px;
    }

    .hero {
        padding: 28px 34px;
        border-radius: 24px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #2563eb 100%);
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.25);
    }

    .hero h1 {
        font-size: 38px;
        margin-bottom: 6px;
        font-weight: 850;
    }

    .hero p {
        font-size: 17px;
        color: #dbeafe;
        margin: 0;
    }

    .section-title {
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 14px;
    }

    .soft-card {
        padding: 20px;
        border-radius: 18px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(148,163,184,0.25);
        box-shadow: 0 8px 22px rgba(0,0,0,0.12);
        margin-bottom: 16px;
    }

    .meta-card {
        padding: 16px 18px;
        border-radius: 16px;
        background: rgba(37, 99, 235, 0.12);
        border: 1px solid rgba(37, 99, 235, 0.35);
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .tool-card {
        padding: 16px 18px;
        border-radius: 16px;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.35);
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .warning-card {
        padding: 16px 18px;
        border-radius: 16px;
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.45);
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .badge {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 999px;
        background: #1d4ed8;
        color: white;
        font-weight: 700;
        font-size: 13px;
        margin-right: 6px;
        margin-bottom: 5px;
    }

    .badge-green {
        background: #15803d;
    }

    .badge-orange {
        background: #d97706;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 13px;
    }

    .bar-container {
        width: 100%;
        background-color: rgba(148,163,184,0.25);
        border-radius: 999px;
        height: 14px;
        overflow: hidden;
        margin-bottom: 12px;
    }

    .bar-fill {
        height: 14px;
        border-radius: 999px;
        background: linear-gradient(90deg, #2563eb, #22c55e);
    }
    </style>
    """,
    unsafe_allow_html=True
)


def health_check(api_url: str):
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, {"error": response.text}
    except Exception as e:
        return False, {"error": str(e)}


def get_intents(api_url: str):
    try:
        response = requests.get(f"{api_url}/intents", timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"intents": [], "total": 0}
    except Exception:
        return {"intents": [], "total": 0}


def send_message(api_url: str, message: str, session_id: str):
    payload = {
        "message": message,
        "session_id": session_id,
    }

    response = requests.post(
        f"{api_url}/chat",
        json=payload,
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(response.text)

    return response.json()


def reset_session(api_url: str, session_id: str):
    try:
        requests.delete(f"{api_url}/session/{session_id}", timeout=5)
    except Exception:
        pass


def confidence_label(confidence: float):
    if confidence >= 0.80:
        return "Très fiable"
    if confidence >= 0.50:
        return "Moyenne"
    return "Faible"


def render_intent_distribution(df: pd.DataFrame):
    counts = df["intent"].value_counts()

    for intent, count in counts.items():
        percentage = int((count / counts.sum()) * 100)

        st.markdown(
            f"""
            <div style="margin-bottom: 8px;">
                <b>{intent}</b>
                <span class="small-muted"> — {count} message(s)</span>
                <div class="bar-container">
                    <div class="bar-fill" style="width: {percentage}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if "session_id" not in st.session_state:
    st.session_state.session_id = f"crm-session-{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "analytics" not in st.session_state:
    st.session_state.analytics = []


with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_url = st.text_input("URL de l’API FastAPI", value=API_URL_DEFAULT)

    st.text_input(
        "Session ID",
        value=st.session_state.session_id,
        key="session_id_input",
    )

    st.session_state.session_id = st.session_state.session_id_input

    confidence_threshold = st.slider(
        "Seuil de confiance pour traitement automatique",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.05,
    )

    connected, health_data = health_check(api_url)

    if connected:
        st.success("API connectée")
        st.caption(f"Timestamp : {health_data.get('timestamp', '')}")
    else:
        st.error("API non connectée")
        st.caption(health_data.get("error", ""))

    st.divider()

    if st.button("🔄 Réinitialiser la session", use_container_width=True):
        reset_session(api_url, st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.analytics = []
        st.success("Session réinitialisée")

    st.divider()

    st.markdown("## 🧠 Intents disponibles")

    intents_data = get_intents(api_url)
    intents = intents_data.get("intents", [])

    st.metric("Nombre total", intents_data.get("total", 0))

    with st.expander("Voir la liste des intents"):
        if intents:
            st.dataframe(pd.DataFrame({"Intent": intents}), use_container_width=True)
        else:
            st.info("Aucun intent chargé.")


st.markdown(
    """
    <div class="hero">
        <h1>🤖 Agent MCP CRM</h1>
        <p>Interface professionnelle pour tester la classification d’intentions, le routage MCP et les réponses CRM.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


top_col1, top_col2, top_col3 = st.columns(3)

with top_col1:
    st.metric("Backend", "TF-IDF + Logistic Regression")

with top_col2:
    st.metric("Mode LLM", "Fallback sans Ollama")

with top_col3:
    st.metric("API", "Connectée" if connected else "Déconnectée")


left_col, right_col = st.columns([2.2, 1])


with left_col:
    st.markdown('<div class="section-title">💬 Interaction client</div>', unsafe_allow_html=True)

    examples = [
        "I want to cancel my order number 12345",
        "I want to modify the items in my order",
        "I want to update my shipping address",
        "How much is the cancellation fee?",
        "I want to check my invoice",
        "What payment methods are available?",
    ]

    selected_example = st.selectbox(
        "Exemples adaptés aux intents du modèle",
        [""] + examples,
    )

    col_a, col_b = st.columns([1, 2])

    with col_a:
        send_example = st.button("▶️ Tester l’exemple", use_container_width=True)

    with col_b:
        st.caption("Tu peux choisir un exemple ou écrire ton propre message client.")

    custom_message = st.text_area(
        "Message client personnalisé",
        placeholder="Exemple : I want to cancel my order number 12345",
        height=100,
    )

    col_send, col_clear = st.columns([1, 1])

    with col_send:
        send_custom = st.button("🚀 Analyser le message", use_container_width=True)

    with col_clear:
        clear_input = st.button("🧹 Nouveau test", use_container_width=True)

    user_input = None

    if send_example and selected_example:
        user_input = selected_example

    if send_custom and custom_message.strip():
        user_input = custom_message.strip()

    if clear_input:
        st.session_state.messages = []
        st.session_state.analytics = []
        st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and "metadata" in msg:
                metadata = msg["metadata"]
                intent = metadata.get("intent", "N/A")
                confidence = float(metadata.get("confidence", 0))
                tool_called = metadata.get("tool_called", "N/A")

                st.markdown(
                    f"""
                    <div class="meta-card">
                        <span class="badge">Intent : {intent}</span>
                        <span class="badge badge-green">Confiance : {confidence:.2%}</span>
                        <span class="badge badge-orange">Niveau : {confidence_label(confidence)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.progress(min(confidence, 1.0))

                st.markdown(
                    f"""
                    <div class="tool-card">
                        <b>Tool MCP appelé :</b> {tool_called}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if confidence < confidence_threshold:
                    st.markdown(
                        """
                        <div class="warning-card">
                            ⚠️ Confiance faible : il est recommandé de rediriger vers un agent humain.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with st.expander("Voir le résultat technique du tool"):
                    st.json(metadata.get("tool_result", {}))

    if user_input:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyse du message client en cours..."):
                try:
                    result = send_message(api_url, user_input, st.session_state.session_id)

                    response_text = result.get("response", "")
                    intent = result.get("intent", "unknown")
                    confidence = float(result.get("confidence", 0.0))
                    tool_called = result.get("tool_called", "none")
                    tool_result = result.get("tool_result", {})

                    st.markdown(response_text)

                    st.markdown(
                        f"""
                        <div class="meta-card">
                            <span class="badge">Intent : {intent}</span>
                            <span class="badge badge-green">Confiance : {confidence:.2%}</span>
                            <span class="badge badge-orange">Niveau : {confidence_label(confidence)}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.progress(min(confidence, 1.0))

                    st.markdown(
                        f"""
                        <div class="tool-card">
                            <b>Tool MCP appelé :</b> {tool_called}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if confidence < confidence_threshold:
                        st.markdown(
                            """
                            <div class="warning-card">
                                ⚠️ Confiance faible : redirection recommandée vers un agent humain.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with st.expander("Voir le résultat technique du tool"):
                        st.json(tool_result)

                    metadata = {
                        "intent": intent,
                        "confidence": confidence,
                        "tool_called": tool_called,
                        "tool_result": tool_result,
                    }

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response_text,
                            "metadata": metadata,
                        }
                    )

                    st.session_state.analytics.append(
                        {
                            "message": user_input,
                            "intent": intent,
                            "confidence": confidence,
                            "tool_called": tool_called,
                        }
                    )

                except Exception as e:
                    error_message = f"Erreur pendant l’appel API : {e}"
                    st.error(error_message)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                        }
                    )


with right_col:
    st.markdown('<div class="section-title">📊 Dashboard</div>', unsafe_allow_html=True)

    if len(st.session_state.analytics) == 0:
        st.info("Aucune interaction analysée pour le moment.")
    else:
        df = pd.DataFrame(st.session_state.analytics)

        total_messages = len(df)
        avg_confidence = df["confidence"].mean()
        low_confidence_count = int((df["confidence"] < confidence_threshold).sum())
        unique_intents = df["intent"].nunique()

        kpi1, kpi2 = st.columns(2)

        with kpi1:
            st.metric("Messages", total_messages)
            st.metric("Intents", unique_intents)

        with kpi2:
            st.metric("Confiance moyenne", f"{avg_confidence:.2%}")
            st.metric("Escalades", low_confidence_count)

        st.divider()

        st.markdown("### Répartition des intents")
        render_intent_distribution(df)

        st.divider()

        st.markdown("### Historique technique")
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Télécharger l’historique CSV",
            data=csv,
            file_name="crm_agent_session_history.csv",
            mime="text/csv",
            use_container_width=True,
        )