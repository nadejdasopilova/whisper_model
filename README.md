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
    uvicorn whisper:app
```
## Использование
1.	Запустить приложение;
2.	POST-запрос на http://127.0.0.1:8000/convert с аудиофайлом предоставит текстовое содержимое отправленного аудиофайла.
 
## Лицензия
Используется открытая модель машинного обучения - https://huggingface.co/openai/whisper-large-v3.
