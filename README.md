# whisper_model
# Программная инженерия. Сопилова Надежда.
# Автоматическое распознавание речи, вывод в формате "текст".

## Начало работы
Инструкция по настройке окружения для скачивания и запуска проекта локально:
1) Создать виртуальное окружение:
```
    python3 -m venv venv
```
2) Активировать виртуальное окружение:
```
    source venv/bin/activate
```
3) Запуск приложения:
```
    streamlit run whisper.py
```
## Использование
1.	Запустить приложение;
2.	Выбрать источник данных - запись звука / загрузка файла;
3.  Произнесите речь - остановите запись или загрузите аудиофайл формата mp3;
4.  Приложение выводит содержимое аудиофайла текстом на экран.
 
## Лицензия
Используется открытая модель машинного обучения - https://huggingface.co/openai/whisper-large-v3.
