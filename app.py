"""Sentiment Analysis Classifier — Streamlit Application."""

from __future__ import annotations

import sys
import os
import time

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.sentiment_model import get_analyzer, SentimentAnalyzer, DEFAULT_MODEL_NAME
from src.preprocessing import validate_input, clean_text
from utils.visualization import (
    create_probability_chart,
    create_sentiment_gauge,
    create_batch_distribution_chart,
    SENTIMENT_COLORS,
)
from utils.csv_handler import process_csv_batch, dataframe_to_csv_bytes

st.set_page_config(
    page_title="Sentiment Analysis Classifier",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    .main-header {
        text-align: center;
        padding: 2rem 1rem 1.5rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.25);
    }
    .main-header h1 {
        color: white;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #e2e8f0;
        font-size: 0.95rem;
        margin: 0;
    }
    .sentiment-card {
        padding: 2rem 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .sentiment-positive {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 2px solid #22c55e;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 2px solid #ef4444;
    }
    .sentiment-neutral {
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
        border: 2px solid #6b7280;
    }
    .sentiment-label {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .sentiment-confidence {
        font-size: 1.3rem;
        margin-top: 0.6rem;
        font-weight: 500;
        color: #475569;
    }
    .info-card {
        background: #f8fafc;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #475569;
    }
    .stat-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin: 0.3rem 0;
    }
    .stat-value { font-size: 2rem; font-weight: 700; color: #667eea; }
    .stat-label { font-size: 0.8rem; color: #94a3b8; margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 0.5px; }
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #94a3b8;
        font-size: 0.85rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
    }
    .footer a { color: #667eea; text-decoration: none; }
    .stButton > button {
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    section[data-testid="stSidebar"] { background: #fafbfc; }
    .stRadio > label { font-weight: 600; color: #334155; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


EXAMPLE_TEXTS = [
    ("Positive Review", "This product is absolutely amazing! The quality exceeds my expectations and the customer service was outstanding. I would highly recommend it to everyone."),
    ("Negative Review", "Terrible experience. The product broke after just two days and the company refused to refund my money. Very disappointed with this purchase."),
    ("Mixed Feedback", "The food was decent but the service was quite slow. The atmosphere was nice though. Overall an okay experience, nothing special."),
    ("Product Question", "Does this laptop come with a warranty? I'm considering buying it for work and want to make sure it's covered."),
]


@st.cache_resource
def load_model() -> SentimentAnalyzer:

    analyzer = get_analyzer()
    analyzer.load()
    return analyzer


def render_header():
    st.markdown(
        """
        <div class="main-header">
            <h1>💬 Sentiment Analysis Classifier</h1>
            <p>Powered by DistilBERT Transformer · NLP · Hugging Face</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown("### 📋 Navigation")
        page = st.radio(
            "Go to",
            ["Single Analysis", "Batch Analysis", "Model Info", "About"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        show_probabilities = st.checkbox("Show class probabilities", value=True)
        show_gauge = st.checkbox("Show confidence gauge", value=True)

        st.markdown("---")
        st.markdown("### 📊 Model Status")
        try:
            with st.spinner("Loading model..."):
                analyzer = load_model()
            st.success("✅ Model loaded")
            st.caption(f"Model: `{DEFAULT_MODEL_NAME}`")
        except Exception as exc:
            st.error(f"❌ Model load failed: {exc}")
            analyzer = None

        return page, show_probabilities, show_gauge, analyzer


def render_sentiment_card(label: str, confidence: float):
    css_class = {
        "POSITIVE": "sentiment-positive",
        "NEGATIVE": "sentiment-negative",
        "NEUTRAL": "sentiment-neutral",
    }.get(label, "sentiment-neutral")

    emoji = {
        "POSITIVE": "😊",
        "NEGATIVE": "😞",
        "NEUTRAL": "😐",
    }.get(label, "😐")

    st.markdown(
        f"""
        <div class="sentiment-card {css_class}">
            <p class="sentiment-label">{emoji} {label}</p>
            <p class="sentiment-confidence">Confidence: {confidence * 100:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_single_analysis(analyzer, show_probabilities, show_gauge):
    st.markdown("## 🔍 Single Text Analysis")
    st.markdown("Enter text below to analyze its sentiment.")

    col_examples = st.columns(4)
    for i, (name, text) in enumerate(EXAMPLE_TEXTS):
        with col_examples[i]:
            if st.button(name, key=f"example_{i}", use_container_width=True):
                st.session_state["input_text"] = text

    default_text = st.session_state.get("input_text", "")
    user_input = st.text_area(
        "Enter text to analyze:",
        value=default_text,
        height=150,
        placeholder="Type or paste your text here...",
        key="text_input",
    )

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
    with col_btn1:
        analyze_clicked = st.button("🚀 Analyze", type="primary", use_container_width=True)
    with col_btn2:
        clear_clicked = st.button("🗑️ Clear", use_container_width=True)

    if clear_clicked:
        st.session_state["input_text"] = ""
        st.rerun()

    if analyze_clicked:
        is_valid, error_msg = validate_input(user_input)
        if not is_valid:
            st.warning(error_msg)
            return

        if analyzer is None:
            st.error("Model is not loaded. Please check the sidebar for errors.")
            return

        with st.spinner("Analyzing sentiment..."):
            try:
                result = analyzer.predict(user_input)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return

        st.markdown("---")
        col_result, col_viz = st.columns([1, 2])

        with col_result:
            render_sentiment_card(result.label, result.confidence)
            st.markdown(
                f"""
                <div class="info-card">
                    <b>Raw model label:</b> {result.raw_label}<br>
                    <b>Raw confidence:</b> {result.raw_confidence * 100:.2f}%
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_viz:
            if show_gauge:
                st.plotly_chart(
                    create_sentiment_gauge(result.label, result.confidence),
                    use_container_width=True,
                )
            if show_probabilities and result.probabilities:
                st.plotly_chart(
                    create_probability_chart(result.probabilities),
                    use_container_width=True,
                )

        with st.expander("📝 Cleaned Text"):
            st.code(clean_text(user_input), language="text")


def page_batch_analysis(analyzer):
    st.markdown("## 📂 Batch Analysis")
    st.markdown("Upload a CSV file with text entries to analyze in bulk.")

    st.info(
        "Upload a CSV file. The application will automatically detect the text column "
        "(first string column), or you can specify it manually."
    )

    col_upload, col_col = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="The CSV should contain at least one column with text data.",
        )
    with col_col:
        col_name = st.text_input(
            "Text column name (optional)",
            placeholder="auto-detect",
            help="Leave empty to auto-detect the first text column.",
        )

    if uploaded_file is not None:
        if analyzer is None:
            st.error("Model is not loaded. Please check the sidebar.")
            return

        with st.spinner("Processing batch..."):
            try:
                result_df = process_csv_batch(
                    uploaded_file,
                    analyzer,
                    text_column=col_name if col_name.strip() else None,
                )
            except ValueError as exc:
                st.error(f"Input error: {exc}")
                return
            except Exception as exc:
                st.error(f"Processing failed: {exc}")
                return

        st.success(f"✅ Processed {len(result_df)} entries.")

        col_chart, col_stats = st.columns([1, 1])
        with col_chart:
            st.plotly_chart(
                create_batch_distribution_chart(result_df["sentiment"].tolist()),
                use_container_width=True,
            )
        with col_stats:
            st.markdown("### Summary")
            counts = result_df["sentiment"].value_counts()
            for label, count in counts.items():
                color = SENTIMENT_COLORS.get(label, "#3b82f6")
                st.markdown(
                    f"<div style='padding:0.5rem 1rem;border-left:4px solid {color};"
                    f"background:#f8fafc;border-radius:4px;margin:0.3rem 0;'>"
                    f"<b>{label}</b>: {count} ({count / len(result_df) * 100:.1f}%)"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("### Results")
        st.dataframe(result_df, use_container_width=True, height=400)

        csv_bytes = dataframe_to_csv_bytes(result_df)
        st.download_button(
            "⬇️ Download Results CSV",
            data=csv_bytes,
            file_name="sentiment_results.csv",
            mime="text/csv",
        )


def page_model_info():
    st.markdown("## 🧠 Model Information")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>Architecture</h3>
                <ul>
                    <li><b>Base Model:</b> DistilBERT</li>
                    <li><b>Parameters:</b> ~67M</li>
                    <li><b>Layers:</b> 6 Transformer layers</li>
                    <li><b>Hidden Size:</b> 768</li>
                    <li><b>Attention Heads:</b> 12</li>
                    <li><b>Vocab Size:</b> 30,522</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h3>Training</h3>
                <ul>
                    <li><b>Dataset:</b> SST-2 (Stanford Sentiment Treebank)</li>
                    <li><b>Task:</b> Binary sequence classification</li>
                    <li><b>Fine-tuned:</b> Yes (by Hugging Face)</li>
                    <li><b>Labels:</b> POSITIVE, NEGATIVE</li>
                    <li><b>Neutral Class:</b> Derived (confidence < 55%)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="info-card">
            <h3>How Neutral Sentiment Works</h3>
            <p>The base DistilBERT model is trained for binary classification
            (Positive / Negative). When the model's confidence in either class
            falls below a <b>55% threshold</b>, the text is classified as
            <b>Neutral</b>. This approach avoids loading a heavier 3-class model
            while still providing practical neutral detection.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_about():
    st.markdown("## ℹ️ About This Application")

    st.markdown(
        """
        <div class="info-card">
            <h3>Sentiment Analysis Classifier</h3>
            <p>An interactive NLP application that uses a pretrained DistilBERT
            transformer model to classify the sentiment of user-provided text
            as <b>Positive</b>, <b>Negative</b>, or <b>Neutral</b>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🛠️ Technologies Used")
    techs = [
        "Python 3.10+",
        "Streamlit — Web UI framework",
        "Hugging Face Transformers — Model library",
        "PyTorch — Deep learning backend",
        "Plotly — Interactive visualizations",
        "Pandas — Data handling",
    ]
    for tech in techs:
        st.markdown(f"- {tech}")

    st.markdown("### 👤 Author")
    st.markdown("**Sawaira Saeed**")

    st.markdown("---")
    st.markdown(
        "<div class='footer'>Sentiment Analysis Classifier · "
        "Powered by Hugging Face Transformers · "
        "Built with Streamlit</div>",
        unsafe_allow_html=True,
    )


def main():
    render_header()
    page, show_probs, show_gauge, analyzer = render_sidebar()

    if page == "Single Analysis":
        page_single_analysis(analyzer, show_probs, show_gauge)
    elif page == "Batch Analysis":
        page_batch_analysis(analyzer)
    elif page == "Model Info":
        page_model_info()
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()