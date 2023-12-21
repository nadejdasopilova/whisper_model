import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from audiorecorder import audiorecorder


@st.cache_resource
def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-small"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )


def show_audio_and_button(audio):
    # выводим воспроизведение аудио
    st.audio(audio)
    # выводим кнопку "Конвертировать"
    return st.button("Конвертировать")


def main():
    # загружаем предварительно обученную модель
    whisper = load_model()
    created_button = False
    audio = None

    st.title("Конвертация аудио в текст")
    st.write("Вы можете использовать аудио на любом из 99 языков")

    # выбор источника данных
    source_button = st.radio(
        "Выберите источник данных",
        ["Запись звука", "Загрузка файла"],
        captions=["Произнесите речь", "Загрузить аудиофайл формата mp3"], index=None
    )
    # запись звука
    if source_button == "Запись звука":
        # чтение аудио с микрофона
        audio = audiorecorder("Записать", "Остановить запись")
        if len(audio) > 0:
            audio = audio.export().read()
            created_button = show_audio_and_button(audio=audio)

    elif source_button == "Загрузка файла":
        # форма для загрузки аудиофайла
        uploaded_file = st.file_uploader("Выберите файл", type="mp3", accept_multiple_files=False)
        if uploaded_file is not None:
            # чтение аудио из файла
            audio = uploaded_file.read()
            created_button = show_audio_and_button(audio=audio)
        else:
            audio = None

    if created_button and audio:
        try:
            # выводим результат
            st.markdown("**Результат:**")
            st.write(whisper(audio)["text"])
        except Exception as e:
            # выводим возникающие ошибки
            st.write(f"Ошибка: {e}")


main()
