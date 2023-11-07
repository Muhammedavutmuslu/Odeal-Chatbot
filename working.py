import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, session
import pickle
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

app.secret_key = '43324'

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))


@app.route('/')
def home():
    # Initialize an empty list to store chat messages
    session['messages'] = []
    return render_template('index.html', messages=[])


@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    user_input = user_input.lower()

    # Kullanıcı girdisini vektörleme işleminden geçirin
    user_input_array = vectorizer.transform([user_input])

    # Modeli kullanarak tahmin yapın
    prediction = model.predict(user_input_array)

    # Kullanıcı mesajını ve bot yanıtını ekleyin
    add_message("user", user_input)
    add_message("bot", prediction[0])

    return render_template("index.html", messages=session['messages'])


def add_message(type, content):
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M")  # Saat ve dakika formatı
    message = {'type': type, 'content': content, 'timestamp': timestamp}
    messages = session.get('messages', [])
    messages.append(message)
    session['messages'] = messages


if __name__ == "__main__":
    app.run(debug=True)


############################


<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>chatbot.com</title>
    <link rel="shortcut icon" href="#">
    <meta http-equiv="cache-control" content="max-age=0">
    <meta http-equiv="cache-control" content="no-cache">
    <meta http-equiv="expires" content="0">
    <meta http-equiv="expires" content="Tue, 01 Jan 1980 1:00:00 GMT">
    <meta http-equiv="pragma" content="no-cache">
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            margin: 0;
            padding: 0;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            padding: 10px;
        }
        .chat-container {
            background: #f9f9f9;
            border: 1px solid #ccc;
            border-radius: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-top: 20px;
            max-height: 600px; /* Set a maximum height */
            overflow-y: auto; /* Enable vertical scrolling if content overflows */
        }
        .message {
            margin-bottom: 70px;
            display: flex;
            justify-content: flex-start;
            align-items: center;
        }
        .avatar {
            width: 40px;
            height: 40px;
            background-color: #fff;
            border-radius: 50%;
            overflow: hidden;
        }
        .avatar img {
            width: 100%;
            height: 100%;
        }
        .message-content {
            background-color: #fff;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .user-message {
            background-color: #007bff;
            color: #fff;
            margin-left: auto; /* Move user messages to the right */
        }
        .bot-message {
            background-color: #ccc;
            color: #333;
            margin-right: auto; /* Move bot messages to the right */
        }
        .timestamp {
            font-size: 15px;
            color: #888;
            margin-left: 1px;
            margin-bottom:5px
           
        }
        form {
            padding: 10px;
            border-top: 1px solid #ccc;
            display: flex;
            align-items: center;
        }
        label {
            font-size: 16px;
            margin-right: 10px;
        }
        input[type="text"] {
            width: 80%;
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        button[type="submit"] {
            padding: 10px 20px;
            font-size: 18px;
            background-color: #007bff;
            color: #fff;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
    </style>
</head>
<body>

    <div class="container">
        <div class="chat-container" id="chat-container">
            {% for message in messages %}
                {% if message.type == 'user' %}
                
                    <div class="message">
                    <div class="timestamp">
                        {{ message.timestamp }}
                    </div>
                        <div class="message-content user-message">
                            {{ message.content }}
                        </div>
                        
                        <div class="avatar">
                            <img src="https://cdn.chatbot.com/widget/61f28451fdd7c5000728b4f9/FPBAPaZFOOqqiCbV.png" alt="Avatar">
                            
                        </div>
                        
                    </div>
                {% elif message.type == 'bot' %}
                    <div class="message">
                        <div class="avatar">
                            <img src="https://www.pngkit.com/png/detail/408-4088709_277kib-1024x576-steve-takes-a-selfie-by-shrewbiez.png">
                        </div>
                        <div class="message-content bot-message">
                            {{ message.content }}
                        </div>
                        
                    </div>
                {% endif %}
            {% endfor %}
        </div>

        <form method="POST" action="/predict">
            <label for="user_input">Type your message here:</label>
            <input type="text" id="user_input" name="user_input" required>
            <button type="submit">Send</button>
        </form>
    </div>

    <script>
        // Scroll to the bottom of the chat container
        function scrollToBottom() {
            var chatContainer = document.getElementById("chat-container");
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        // Call scrollToBottom when the page loads to ensure it starts at the bottom
        window.onload = scrollToBottom;
    </script>
</body>
</html>

