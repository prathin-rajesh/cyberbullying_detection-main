from flask import Flask, render_template, request
import torch
import pickle
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)

cred = credentials.Certificate("cyberbullyingdetails-firebase-admin.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "tinybert_model.pkl"
tokenizer_path = "tinybert_tokenizer.pkl"

# Load the trained TinyBERT model
model = AutoModelForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", num_labels=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load the tokenizer
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

def load_profane_words(file_path="profane_words.txt"):
    """Loads profane words from a text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return set(word.strip().lower() for word in f.readlines())

profane_words = load_profane_words()

def censor_word(word):
    """Censors a profane word by replacing middle characters with asterisks."""
    if len(word) > 2:
        return word[0] + "*" * (len(word) - 2) + word[-1]
    return word  # If the word is too short, keep it the same

def cleanse_text(text):
    """Replaces profane words with censored versions."""
    words = text.split()  # Split sentence into words
    censored_words = [censor_word(word) if word.lower() in profane_words else word for word in words]
    return " ".join(censored_words)  # Rejoin the words into a sentence

def contains_profanity(text):
    """Returns True if text contains profanity, else False."""
    words = text.lower().split()
    return any(word in profane_words for word in words)

def predict_cyberbullying(text):
    print(f"User Input: {text}")  # Debugging statement

    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    
    print(f"Model Prediction: {prediction}, Logits: {logits}")  # Debugging output
    return "Cyberbullying" if prediction == 1 else "Not Cyberbullying"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    cleansed_text = None  # Store the censored version

    if request.method == 'POST':
        user_input = request.form['text']

        # Check for profanity and cleanse if necessary
        if contains_profanity(user_input):
            cleansed_text = cleanse_text(user_input)

        prediction = predict_cyberbullying(user_input)

    return render_template('index.html', prediction=prediction, cleansed_text=cleansed_text)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    success = False
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        feedback = request.form['feedback']

        # Store data in Firestore
        contact_ref = db.collection("contacts")
        contact_ref.add({
            "name": name,
            "email": email,
            "feedback": feedback
        })

        success = True

    return render_template('contact.html', success=success)

if __name__ == '__main__':
    app.run(debug=True)
