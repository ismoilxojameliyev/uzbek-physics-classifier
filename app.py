import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Uzbek Physics AI", page_icon="⚛️")

st.title("⚛️ Uzbek Physics AI Classifier")
st.write("Ushbu sun'iy intellekt kiritilgan matnning fizika masalasi ekanligini avtomatik aniqlaydi.")

# Hugging Face model repository path
MODEL_NAME = "ismoilxojameliyev/uzbek-physics-model"

@st.cache_resource
def load_classifier():
    return pipeline("text-classification", model=MODEL_NAME)

try:
    classifier = load_classifier()
except Exception as e:
    classifier = None
    st.error(f"Modelni yuklashda xatolik: {e}. Model Hugging Face-da mavjudligini tekshiring.")

text_input = st.text_area(
    "Matnni kiriting:",
    placeholder="Masalan: 10 sm masofada turgan ikkita zaryadlangan sharcha...",
    height=150
)

if st.button("Tahlil qilish (Analyze)"):
    if not text_input.strip():
        st.warning("Iltimos, avval matn kiriting.")
    elif classifier is None:
        st.error("Model hali to'liq ulanmagan.")
    else:
        with st.spinner("Tahlil qilinmoqda..."):
            result = classifier(text_input)[0]
            label = str(result["label"])
            score = round(result["score"] * 100, 1)

            if label in ["LABEL_1", "1"]:
                st.success(f"✅ **Fizika masalasi** (Ishonch darajasi: {score}%)")
            else:
                st.info(f"❌ **Boshqa matn / boshqa fan** (Ishonch darajasi: {score}%)")
