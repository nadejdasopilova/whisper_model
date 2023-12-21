from fastapi.testclient import TestClient
from src.api.whisper_api import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == "\"A\'m alive\""


def test_convert():
    files = {'audio': ('Sound_08129.mp3', open("test/api/Sound_08129.mp3", 'rb'))}
    response = client.post("/convert", files=files)
    assert response.status_code == 200
    assert response.text == '\" Внимание! Говорит и показывает Москва. Работают все центральные каналы телевидения. Смотрите и слушайте Москву.\"'


def test_convert2():
    files = {'audio': ('Sound_07645.mp3', open("test/api/Sound_07645.mp3", 'rb'))}
    response = client.post("/convert", files=files)
    assert response.status_code == 200
    assert response.text == '\" О нашем времени, о наших подвигах будут всегда помнить люди. Пройдут столетия, и наши правдоки будут рассказывать, как герои с ним боролись, как умели умирать за свободу, за благо народа. О нас будут все песни, расплавляя нашу борьбу. Забудут отдельных лиц Калинина, Петрова, Иванова. Но всех вместе будут вспоминать половинам и божищу.\"'


def test_convert_422():
    files = {'broken_name': ('Sound_07645.mp3', open("test/api/Sound_07645.mp3", 'rb'))}
    response = client.post("/convert", files=files)
    assert response.status_code == 422
