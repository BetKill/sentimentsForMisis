import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

# Указываем устройство — обязательно CPU для Streamlit Cloud
device = torch.device("cpu")

# Загрузка модели и токенизатора с Hugging Face
model_id = "BetKill1994/diploms"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
model.eval()

# Интерфейс
st.title("Анализ тональности текста")
text = st.text_area("Введите текст для анализа", "")

if st.button("Анализировать") and text:
    # Токенизация текста
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    # Перенос входных данных на устройство (CPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            output = model(**inputs)
            logits = output.logits
            probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()

        labels = ["Нейтральный", "Позитивный", "Негативный"]
        predicted = labels[torch.argmax(logits)]

        st.markdown(f"**Предсказанная тональность:** {predicted}")

        # Отображение гистограммы вероятностей
        fig, ax = plt.subplots()
        ax.bar(labels, probs, color=["gray", "green", "red"])
        ax.set_ylabel("Вероятность")
        ax.set_title("Распределение вероятностей по классам")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Произошла ошибка: {e}")
