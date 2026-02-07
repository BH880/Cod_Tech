import random
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ----------------------------------------
# LOAD NLP MODELS (Python 3.12 Compatible)
# ----------------------------------------
try:
   nlp = spacy.load("en_core_web_md")
except OSError:
    # If model not found, download automatically
    from spacy.cli import download
    download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

# Download NLTK punkt tokenizer
nltk.download("punkt")

# ----------------------------------------
# TRAINING DATA (INTENTS)
# ----------------------------------------
training_sentences = [
    "hello", "hi", "good morning", "hey there", "good evening",
    "what is your name", "who are you",
    "how are you", "how is it going",
    "what is NLP", "explain nlp", "define natural language processing",
    "what is machine learning", "define ml",
    "bye", "goodbye", "see you later"
]

training_labels = [
    "greeting", "greeting", "greeting", "greeting", "greeting",
    "bot_identity", "bot_identity",
    "bot_status", "bot_status",
    "nlp_def", "nlp_def", "nlp_def",
    "ml_def", "ml_def",
    "goodbye", "goodbye", "goodbye"
]

# ----------------------------------------
# ML MODEL FOR INTENT CLASSIFICATION
# ----------------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_sentences)

model = LogisticRegression()
model.fit(X, training_labels)

# ----------------------------------------
# INTENT RESPONSES
# ----------------------------------------
responses = {
    "greeting": [
        "Hello! How can I assist you today?",
        "Hi there! What can I do for you?",
        "Greetings! How may I help?"
    ],
    "bot_identity": [
        "I am an NLP-powered AI chatbot.",
        "I'm an intelligent assistant built using Python NLP!",
    ],
    "bot_status": [
        "I'm functioning perfectly!",
        "Doing great! How can I help you?",
    ],
    "nlp_def": [
        "NLP stands for Natural Language Processing — the field of AI that helps computers understand human language.",
        "NLP enables machines to read, understand, and generate human language.",
    ],
    "ml_def": [
        "Machine Learning is a branch of AI that enables systems to learn patterns from data.",
        "ML allows computers to improve performance using experience—data!",
    ],
    "goodbye": [
        "Goodbye! Have a great day!",
        "See you later!",
    ]
}

# ----------------------------------------
# SEMANTIC SIMILARITY FALLBACK
# ----------------------------------------
def semantic_similarity(query, known_sentences):
    doc1 = nlp(query)
    best_score = 0
    best_match = None
    
    for sent in known_sentences:
        doc2 = nlp(sent)
        score = doc1.similarity(doc2)
        if score > best_score:
            best_score = score
            best_match = sent
    
    return best_match, best_score

# ----------------------------------------
# CHATBOT ENGINE
# ----------------------------------------
def reply(user_input):
    # ---------- Intent Classification ----------
    X_input = vectorizer.transform([user_input])
    predicted_intent = model.predict(X_input)[0]

    # If confidence is high, use intent-based answer
    confidence = model.predict_proba(X_input).max()
    if confidence > 0.55:
        return random.choice(responses[predicted_intent])

    # ---------- Semantic Similarity Fallback ----------
    best_sentence, score = semantic_similarity(user_input, training_sentences)
    if score > 0.70:
        best_intent = training_labels[training_sentences.index(best_sentence)]
        return random.choice(responses[best_intent])

    # ---------- Named Entity Recognition ----------
    doc = nlp(user_input)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    if entities:
        return f"I found these entities in your message: {entities}"

    # ---------- Final fallback ----------
    return "I'm not sure how to answer that, but I'm learning every day!"

# ----------------------------------------
# CHAT LOOP
# ----------------------------------------
print("AI Chatbot Started! Type 'quit' to exit.\n")

while True:
    user = input("You: ")
    if user.lower() in ["quit", "exit", "bye"]:
        print("Bot:", random.choice(responses["goodbye"]))
        break
    print("Bot:", reply(user))
