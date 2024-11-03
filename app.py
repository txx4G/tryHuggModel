import telebot
from telebot import types
import os
from transformers import VitsModel, AutoTokenizer
import torch
import scipy

# Создаем бота
bot = telebot.TeleBot('7964086337:AAHsgPIhcxZCa-bG_TU3-s4KJ1hHO2fQ2jk')

# Путь к файлу голосового примера
audio_folder = 'audio_messages'
example_audio_file = os.path.join(audio_folder, '/examples/1.wav')

# Инициализация модели и токенизатора
model = VitsModel.from_pretrained("joefox/tts_vits_ru_hf")
tokenizer = AutoTokenizer.from_pretrained("joefox/tts_vits_ru_hf")

# Храним состояния и выбранные параметры для каждого пользователя
user_data = {}

# Команда /start
@bot.message_handler(commands=['start'])
def start(message):
    user_id = message.chat.id
    user_data[user_id] = {'text': None, 'genre': None, 'voice': None, 'mood': None}

    # Приветственное сообщение
    bot.send_message(user_id, "Привет! Я бот для преобразования текста в вокал. Введите текст песни:")

    # Добавляем кнопки
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    example_button = types.KeyboardButton("Пример")
    help_button = types.KeyboardButton("Справка")
    markup.add(example_button, help_button)
    bot.send_message(user_id, "Нажмите на кнопку, чтобы получить помощь или прослушать образец.", reply_markup=markup)

# Обработчик для кнопки "Пример"
@bot.message_handler(func=lambda message: message.text == "Пример")
def send_example_audio(message):
    # Проверяем, существует ли файл примера
    if os.path.exists(example_audio_file):
        with open(example_audio_file, 'rb') as audio:
            bot.send_voice(message.chat.id, audio)
    else:
        bot.send_message(message.chat.id, "Голосовое сообщение с примером не найдено.")

# Обработчик для кнопки "Справка"
@bot.message_handler(func=lambda message: message.text == "Справка")
def send_help(message):
    help_text = (
        "Это бот для преобразования текста в вокал.\n"
        "1. Введите текст песни.\n"
        "2. Выберите жанр: Поп, Рок или Джаз.\n"
        "3. Выберите голос: Мужской или Женский.\n"
        "4. Выберите настроение: Веселое, Грустное или Нейтральное.\n"
        "5. Нажмите 'Начать создание вокала', чтобы получить голосовое сообщение.\n"
        "Для прослушивания примера нажмите 'Пример'."
    )
    bot.send_message(message.chat.id, help_text)

# Получение текста песни
@bot.message_handler(func=lambda message: user_data.get(message.chat.id, {}).get('text') is None)
def get_text(message):
    user_data[message.chat.id]['text'] = message.text
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    markup.add('Поп', 'Рок', 'Джаз')
    bot.send_message(message.chat.id, "Выберите жанр:", reply_markup=markup)

# Выбор жанра
@bot.message_handler(func=lambda message: user_data.get(message.chat.id, {}).get('genre') is None)
def get_genre(message):
    genre = message.text
    if genre not in ['Поп', 'Рок', 'Джаз']:
        bot.send_message(message.chat.id, "Неверный жанр. Пожалуйста, выберите из предложенных вариантов: Поп, Рок или Джаз.")
    else:
        user_data[message.chat.id]['genre'] = genre
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
        markup.add('Мужской', 'Женский')
        bot.send_message(message.chat.id, "Выберите голос:", reply_markup=markup)

# Выбор голоса
@bot.message_handler(func=lambda message: user_data.get(message.chat.id, {}).get('voice') is None)
def get_voice(message):
    voice = message.text
    if voice not in ['Мужской', 'Женский']:
        bot.send_message(message.chat.id, "Неверный голос. Пожалуйста, выберите из предложенных вариантов: Мужской или Женский.")
    else:
        user_data[message.chat.id]['voice'] = voice
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
        markup.add('Веселое', 'Грустное', 'Нейтральное')
        bot.send_message(message.chat.id, "Выберите настроение:", reply_markup=markup)

# Выбор настроения
@bot.message_handler(func=lambda message: user_data.get(message.chat.id, {}).get('mood') is None)
def get_mood(message):
    mood = message.text
    if mood not in ['Веселое', 'Грустное', 'Нейтральное']:
        bot.send_message(message.chat.id, "Неверное настроение. Пожалуйста, выберите из предложенных вариантов: Веселое, Грустное или Нейтральное.")
    else:
        user_data[message.chat.id]['mood'] = mood
        user_data_complete = all(user_data[message.chat.id].values())

        if user_data_complete:
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            create_voice_button = types.KeyboardButton("Начать создание вокала")
            markup.add(create_voice_button)
            bot.send_message(message.chat.id, "Все параметры выбраны. Нажмите 'Начать создание вокала', чтобы получить голосовое сообщение.", reply_markup=markup)

# Обработчик для кнопки "Начать создание вокала"
@bot.message_handler(func=lambda message: message.text == "Начать создание вокала")
def create_voice(message):
    chat_id = message.chat.id
    bot.send_message(chat_id, "Начинаю процесс создания вокала...")
    generate_voice(chat_id)  # Вызов функции генерации вокала

# Функция для генерации вокала
def generate_voice(chat_id):
    text = user_data[chat_id]['text']
    text = text.lower()
    inputs = tokenizer(text, return_tensors="pt")
    inputs['speaker_id'] = 3  # Замените на ID выбранного говорящего при необходимости

    with torch.no_grad():
        output = model(**inputs).waveform

    output_file = "output_voice.wav"
    scipy.io.wavfile.write(output_file, rate=model.config.sampling_rate, data=output[0].cpu().numpy())

    with open(output_file, 'rb') as audio:
        bot.send_voice(chat_id, audio)

# Запуск бота
bot.polling()
