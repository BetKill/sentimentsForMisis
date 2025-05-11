import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt


# Предзагрузка классов torch (важно для DirectML и некоторых сред)
_ = torch.classes

# Загрузка модели
model_id = "BetKill1994/diploms"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    st.success("Модель успешно загружена")
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")


# Метки классов (важно чтобы соответствовали обучению)
labels = ["Нейтральный", "Позитивный", "Негативный"]

# Интерфейс
st.title("📊 Анализ тональности текста")
text = st.text_area("Введите текст для анализа", "", height=150)

if st.button("🔍 Анализировать") and text.strip():
    with st.spinner("Анализируем..."):
        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            with torch.no_grad():
                output = model(**inputs)
                logits = output.logits
                probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()

            predicted = labels[torch.argmax(logits)]

            st.success(f"**Предсказанная тональность:** {predicted}")

            fig, ax = plt.subplots()
            ax.bar(labels, probs, color=["gray", "green", "red"])
            ax.set_ylabel("Вероятность")
            ax.set_title("Распределение по классам")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Ошибка во время анализа: {e}")
