import re
import os
from tokenizers import ByteLevelBPETokenizer
import json

# -----------------------------
# Настройки
# -----------------------------
CORPUS_FILE = "data/corpus.txt"        # исходный текст (твой 4GB корпус)
SAVE_DIR = "data"                       # куда сохранять файлы
VOCAB_SIZE = 15000                      # словарь токенов
MIN_FREQ = 2                            # минимальная частота токена

os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# Шаг 1: Чистка текста
# -----------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)                   # убираем HTML
    text = re.sub(r"\[[0-9]+\]", " ", text)               # цифры в скобках
    text = re.sub(r"[^а-яА-ЯёЁa-zA-ZәіңғүұқөһӘІҢҒҮҰҚӨҺ\s.,!?-]", " ", text)  # мусор
    text = re.sub(r"\s+", " ", text)
    return text.strip()

print("🔹 Чистим корпус...")
with open(CORPUS_FILE, "r", encoding="utf-8", errors="ignore") as f:
    raw_text = f.read()

clean = clean_text(raw_text)

clean_path = os.path.join(SAVE_DIR, "clean_corpus.txt")
with open(clean_path, "w", encoding="utf-8") as f:
    f.write(clean)
print(f"✅ Чистый корпус сохранён в {clean_path}")

# -----------------------------
# Шаг 2: Обучение BPE-токенизатора
# -----------------------------
print("🔹 Обучаем BPE-токенизатор...")
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=[clean_path],
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQ,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)
tokenizer.save_model(SAVE_DIR, f"kazakh_bpe_{VOCAB_SIZE}")
print(f"✅ Токенизатор обучен и сохранён в {SAVE_DIR}/kazakh_bpe_{VOCAB_SIZE}-*")

# -----------------------------
# Шаг 3: Преобразуем текст в токены
# -----------------------------
print("🔹 Преобразуем текст в токены...")
tokenizer.enable_truncation(max_length=256)  # обрезаем до 256 токенов для GPT-0.5

token_ids = []
with open(clean_path, "r", encoding="utf-8") as f:
    for line in f:
        ids = tokenizer.encode(line).ids
        token_ids.extend(ids)

# Сохраняем массив токенов
tokens_path = os.path.join(SAVE_DIR, "tokens.json")
with open(tokens_path, "w", encoding="utf-8") as f:
    json.dump(token_ids, f)

print(f"✅ Массив токенов сохранён в {tokens_path}")
print(f"Общее количество токенов: {len(token_ids)}")