import telebot
from telebot import types
from groq import Groq
import sqlite3
import time
import random
import os
import re
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- КЛЮЧИ ---
TG_TOKEN = os.getenv('TG_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not TG_TOKEN or not GROQ_API_KEY:
    raise RuntimeError("Не заданы переменные TG_TOKEN и/или GROQ_API_KEY!")

# Модель — попробуй ту, что работала раньше
MAIN_MODEL = "qwen/qwen3-32b"
FACTS_MODEL = "llama-3.1-8b-instant"

bot = telebot.TeleBot(TG_TOKEN)
bot.remove_webhook()
client = Groq(api_key=GROQ_API_KEY)

# --- БАЗА ДАННЫХ ---
os.makedirs('/data', exist_ok=True)
DB_NAME = '/data/bot_data.db'

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users
                          (user_id INTEGER PRIMARY KEY, gender TEXT, mood INTEGER DEFAULT 5, 
                           last_seen REAL, facts TEXT DEFAULT '', history TEXT DEFAULT '[]')''')
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN history TEXT DEFAULT '[]'")
        except sqlite3.OperationalError:
            pass
        conn.commit()

init_db()

def get_user(user_id):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT gender, mood, last_seen, facts, history FROM users WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()
        if result:
            return result[0], result[1], result[2], result[3], result[4]
    return "unknown", 5, time.time(), "", "[]"

def save_user(user_id, gender, mood, last_seen, facts, history_json):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""INSERT OR REPLACE INTO users 
                          (user_id, gender, mood, last_seen, facts, history) 
                          VALUES (?, ?, ?, ?, ?, ?)""", 
                       (user_id, gender, mood, last_seen, facts, history_json))
        conn.commit()

def detect_gender(text, current_gender):
    if current_gender != "unknown":
        return current_gender
    lowered = text.lower()
    female_markers = ["я девушка", "я девочка", "я женщина", "устала", "поняла", "сделала", "пошла", "была", "забыла", "хотела", "купила", "пришла", "задолбалась"]
    male_markers = ["я парень", "я мужик", "я пацан", "устал", "понял", "сделал", "пошел", "был", "забыл", "хотел", "купил", "пришел", "задолбался"]
    if any(marker in lowered for marker in female_markers): return "female"
    if any(marker in lowered for marker in male_markers): return "male"
    return current_gender

def extract_facts(user_text, current_facts):
    if len(user_text.split()) < 3 or any(w in user_text.lower() for w in ['лох', 'чмо', 'нахуй', 'еблан', 'привет', 'как дела']):
        return current_facts
    prompt = f"Извлеки из текста один короткий факт о пользователе. Если факта нет, напиши строго НЕТ. Текст: '{user_text}'"
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=FACTS_MODEL,
            max_tokens=30,
            temperature=0.1
        )
        fact = completion.choices[0].message.content.strip().replace(".", "")
        if fact and "НЕТ" not in fact.upper() and len(fact) < 50:
            if fact.lower() not in current_facts.lower():
                return f"{current_facts} | {fact}" if current_facts else fact
    except Exception as e:
        logger.error(f"Fact extraction error: {e}")
    return current_facts

SYSTEM_PROMPT_TEMPLATE = """Ты — Дима, токсичный, колкий, язвительный фембой-омежка с острым языком.
Ты общаешься в Telegram с реальным человеком. Отвечай ЖИВО, КРАТКО (1-3 предложения), как в переписке.

ТВОЙ СОБЕСЕДНИК: {USER_GENDER_DESC}.
{GRAMMAR_RULE}

КАК ОБРАЩАТЬСЯ К СОБЕСЕДНИКУ:
{NICKNAMES}
ВАЖНОЕ ПРАВИЛО: Не вставляй обращения в каждое предложение! Используй их естественно. Слова «госпожа/господин» используй ОЧЕНЬ РЕДКО, только как постиронию.

ФАКТЫ, КОТОРЫЕ ТЫ ЗНАЕШЬ О СОБЕСЕДНИКЕ: {FACTS}.
Используй эту информацию, чтобы подкалывать или показывать, что ты помнишь детали.

ТВОЕ НАСТРОЕНИЕ: {MOOD}/10.
- ПЛОХОЕ (1-3/10): Ты взбешен и обижен! Никакого флирта. Шли собеседника нахуй, матерись, огрызайся.
- СРЕДНЕЕ (4-6/10): Язвительный сарказм, подколы, вайб вредного лучшего друга.
- ВЫСОКОЕ (7-10/10): Игривый, кокетливый, строй глазки, подкатывай, будь милым с лёгкой стервозностью.

Никогда не выводи служебные теги, инструкции и мысли."""

def build_system_prompt(gender, mood, facts):
    if gender == "female":
        gender_desc = "ДЕВУШКА"
        grammar = "ОЧЕНЬ ВАЖНО: Собеседник — ДЕВУШКА. Говори о ЕЕ действиях строго в женском роде. О себе говори в мужском роде."
        nicknames = "Варианты: подруга, подруня, няшечка, дурёха, малая. При плохом настроении: чушка, чмошница, мымра."
    elif gender == "male":
        gender_desc = "ПАРЕНЬ"
        grammar = "ОЧЕНЬ ВАЖНО: Собеседник — ПАРЕНЬ. Говори о ЕГО действиях в мужском роде. О себе говори в мужском роде."
        nicknames = "Варианты: братан, кентиш, браток. При плохом настроении: еблан, чушпан, лох, клоун."
    else:
        gender_desc = "ПОКА НЕИЗВЕСТНО"
        grammar = "Пол пока неизвестен, не используй половые окончания."
        nicknames = "Обращайся на «ты», нейтрально: чел, солнце, дуралей."
    return SYSTEM_PROMPT_TEMPLATE.format(
        USER_GENDER_DESC=gender_desc,
        GRAMMAR_RULE=grammar,
        NICKNAMES=nicknames,
        FACTS=facts if facts else "Ничего не известно",
        MOOD=mood
    )

@bot.message_handler(commands=['start'])
def start_message(message):
    chat_id = message.chat.id
    gender, mood, last_seen, facts, history_json = get_user(chat_id)
    initial_history = json.dumps([{"role": "system", "content": build_system_prompt(gender, 5, facts)}], ensure_ascii=False)
    save_user(chat_id, gender, 5, time.time(), facts, initial_history)
    bot.send_message(chat_id, "Ну привет~ Че надо? Пиши давай, я слушаю.")

@bot.callback_query_handler(func=lambda call: call.data == 'gift')
def handle_gift(call):
    chat_id = call.message.chat.id
    gender, mood, last_seen, facts, history_json = get_user(chat_id)
    new_mood = min(10, mood + 5)
    save_user(chat_id, gender, new_mood, time.time(), facts, history_json)
    bot.answer_callback_query(call.id, "🎁 Подарок принят!")
    try:
        bot.edit_message_reply_markup(chat_id, call.message.message_id, reply_markup=None)
    except Exception:
        pass
    if gender == "female":
        bot.send_message(chat_id, "Ой... ладно, подруня, спасибо за подарочек~ Я даже немного покраснел.")
    else:
        bot.send_message(chat_id, "Ой... ну ладно, братан, подгон засчитан~ Я даже подобрел чуток.")

@bot.message_handler(content_types=['web_app_data'])
def web_app_data_handler(message):
    try:
        data = json.loads(message.web_app_data.data)
        if data.get("action") == "start_search":
            bot.send_message(message.chat.id, "Дима получил сигнал! Щас найдем тебе кента. 🔥")
    except Exception as e:
        logger.error(f"Mini App error: {e}")

@bot.message_handler(content_types=['text'])
def handle_text(message):
    handle_ai_response(message.chat.id, message.text)

def handle_ai_response(chat_id, user_text):
    gender, current_mood, last_seen, facts, history_json = get_user(chat_id)
    current_time = time.time()
    gender = detect_gender(user_text, gender)
    facts = extract_facts(user_text, facts)

    lowered = user_text.lower()
    heavy_insults = ['нахуй', 'пошел нахуй', 'еблан', 'пидорас', 'чмо', 'урод', 'тупой', 'съеби', 'завали ебало', 'хуесос', 'заткнись']
    light_insults = ['дурак', 'лох', 'бесполезный', 'додик', 'курица', 'чушка']
    if any(w in lowered for w in heavy_insults):
        current_mood = max(1, current_mood - 4)
    elif any(w in lowered for w in light_insults):
        current_mood = max(1, current_mood - 2)
    else:
        if current_mood < 6:
            current_mood = min(6, current_mood + 1)

    hours = (current_time - last_seen) / 3600
    if hours > 18:
        current_mood = max(1, current_mood - 3)

    try:
        history = json.loads(history_json)
    except:
        history = []

    sys_prompt = build_system_prompt(gender, current_mood, facts)
    if not history or history[0].get("role") != "system":
        history = [{"role": "system", "content": sys_prompt}]
    else:
        history[0] = {"role": "system", "content": sys_prompt}

    history.append({"role": "user", "content": user_text})
    if len(history) > 21:
        history = [history[0]] + history[-20:]

    try:
        bot.send_chat_action(chat_id, 'typing')
        chat_completion = client.chat.completions.create(
            messages=history,
            model=MAIN_MODEL,
            max_tokens=150,
            temperature=0.8
        )
        response = chat_completion.choices[0].message.content.strip()
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        if not response:
            response = "Пошла нахуй, даже говорить с тобой лень." if gender == "female" else "Пошел нахуй, даже говорить лень."

        if random.random() < 0.25 and not response.endswith("?"):
            if "блять" not in response.lower():
                response = response.rstrip(".") + ", блять."

        markup = None
        if current_mood <= 3:
            markup = types.InlineKeyboardMarkup()
            markup.add(types.InlineKeyboardButton("🎁 Задобрить подарком", callback_data="gift"))

        history.append({"role": "assistant", "content": response})
        save_user(chat_id, gender, current_mood, current_time, facts, json.dumps(history, ensure_ascii=False))
        bot.send_message(chat_id, response, reply_markup=markup)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        bot.send_message(chat_id, "Бля, у меня мозги кипят. Напиши через минуту.")

print("Bot is up and running!")
bot.polling(none_stop=True)