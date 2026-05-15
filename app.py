from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "output" / "best.pt"
IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]


@st.cache_resource
def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def device_name() -> str | int:
    return 0 if torch.cuda.is_available() else "cpu"


def run_detection(model: YOLO, image: Image.Image, confidence: float):
    return model.predict(
        source=image,
        imgsz=640,
        conf=confidence,
        device=device_name(),
        verbose=False,
    )[0]


def detections_table(result) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        rows.append(
            {
                "classe": result.names.get(class_id, str(class_id)),
                "confianca": round(confidence, 3),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Detector de residuos na natureza", layout="wide")

    st.title("Detector de residuos na natureza")

    if not WEIGHTS_PATH.exists():
        st.error(f"Peso nao encontrado: {WEIGHTS_PATH}")
        st.stop()

    model = load_model(str(WEIGHTS_PATH))

    with st.sidebar:
        confidence = st.slider("Confianca minima", 0.05, 0.95, 0.25, 0.05)
        st.caption(f"Peso: {WEIGHTS_PATH.relative_to(ROOT)}")
        st.caption(f"Dispositivo: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    uploaded_file = st.file_uploader("Imagem", type=IMAGE_TYPES)
    if uploaded_file is None:
        st.info("Envie uma imagem para iniciar a deteccao.")
        return

    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Processando imagem..."):
        result = run_detection(model, image, confidence)

    annotated_image = result.plot()[:, :, ::-1]
    table = detections_table(result)

    image_column, data_column = st.columns([2, 1])

    with image_column:
        st.image(annotated_image, caption="Imagem analisada", use_container_width=True)

    with data_column:
        st.metric("Deteccoes", len(table))
        if table.empty:
            st.warning("Nenhum residuo detectado com a confianca atual.")
        else:
            st.dataframe(table, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
