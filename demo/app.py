import logging

import gradio as gr

from src.config import DEMO_HOST, DEMO_PORT, DEMO_SHARE, MIN_TEXT_LENGTH
from src.models import SentimentAnalyzer, create_analyzer, normalize_label
from src.preprocessing import preprocess_text

logger = logging.getLogger(__name__)

models: dict[str, SentimentAnalyzer] = {}


def load_models():
    models["distilbert"] = create_analyzer("distilbert")
    models["roberta"] = create_analyzer("roberta")
    logger.info("Models loaded: %s", list(models.keys()))


def analyze_sentiment(text: str, model_name: str) -> tuple[str, str, str]:
    if not text or not text.strip():
        return "Enter some text to analyze.", "", ""

    cleaned = preprocess_text(text)
    if len(cleaned.strip()) < MIN_TEXT_LENGTH:
        return "Text too short after cleanup.", "", ""

    result = models[model_name].predict(cleaned)
    sentiment = normalize_label(result.label)

    return (
        sentiment.value.capitalize(),
        f"{result.confidence:.4f}",
        f"{result.inference_time * 1000:.1f} ms",
    )


def build_interface() -> gr.Interface:
    return gr.Interface(
        fn=analyze_sentiment,
        inputs=[
            gr.Textbox(label="Review Text", placeholder="Type a movie review...", lines=4),
            gr.Radio(choices=["distilbert", "roberta"], value="distilbert", label="Model"),
        ],
        outputs=[
            gr.Textbox(label="Sentiment"),
            gr.Textbox(label="Confidence"),
            gr.Textbox(label="Inference Time"),
        ],
        title="IMDB Sentiment Analysis",
        description="Analyze movie review sentiment using DistilBERT or RoBERTa.",
        examples=[
            ["This movie was absolutely fantastic! Great acting and storyline.", "distilbert"],
            ["Terrible film. Waste of time and money.", "roberta"],
            ["The plot was predictable but the performances were solid.", "distilbert"],
        ],
        flagging_mode="never",
    )


if __name__ == "__main__":
    load_models()
    demo = build_interface()
    demo.launch(server_name=DEMO_HOST, server_port=DEMO_PORT, share=DEMO_SHARE)
