#download_madel.py
import os
import urllib.request
import sys

# Ссылка на ту самую модель (IQ3_M), которую мы выбрали
MODEL_URL = "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-IQ3_M.gguf?download=true"
FILENAME = "Qwen2.5-3B-Instruct-IQ3_M.gguf"


def download_model():
    if os.path.exists(FILENAME):
        print(f"✅ Модель {FILENAME} уже скачана.")
        return

    print(f"⏳ Скачиваю модель {FILENAME} (1.5 GB)... Это может занять время.")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = int(100 * downloaded / total_size)
        sys.stdout.write(f"\rСкачано: {percent}% ({downloaded / (1024 * 1024):.1f} MB)")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(MODEL_URL, FILENAME, progress)
        print(f"\n✅ Загрузка завершена! Файл сохранен: {FILENAME}")
    except Exception as e:
        print(f"\n❌ Ошибка при скачивании: {e}")


if __name__ == "__main__":
    download_model()
