from flask import Flask, render_template, request, session, redirect, url_for
from flask_session import Session
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------
# Initialize Flask app
# -------------------------
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# -------------------------
# Load Hadith model and FAISS index
# -------------------------
hadith_model = SentenceTransformer('model_name')  # Replace 'model_name' with your actual model name
faiss_index = faiss.read_index('index_file_path')  # Replace 'index_file_path' with your FAISS index path
hadiths_list = [...]  # Your list of hadiths (texts), loaded from your data

# -------------------------
# University chatbot responses
# -------------------------
responses = {
    "greeting": "Hi there! Welcome to your University Admission Assistant!",
    "apply": "To apply for admission, visit: https://youruniversity.edu/apply",
    "requirements": "Requirements:<br>Undergraduate: 50% in Intermediate.<br>Graduate: 2.5 CGPA & test/interview.",
    "dates": "Important Dates:<br>Deadline: 30th June<br>Entry Test: 10th July<br>Merit List: 15th July<br>Classes: 1st August",
    "contact": "Contact Us:<br>Phone: +92-300-0000000<br>Email: admissions@youruniversity.edu",
    "scholarships": "Scholarships:<br>Merit-based, Need-based, Talent-based.<br>Visit: https://youruniversity.edu/scholarships"
}

# -------------------------
# Helper function: Search Hadiths
# -------------------------
def search_hadiths(query, top_k=3):
    query_embedding = hadith_model.encode([query])
    D, I = faiss_index.search(np.array(query_embedding).astype('float32'), top_k)
    results = [hadiths_list[idx] for idx in I[0] if idx != -1]
    return results

# -------------------------
# Flask Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []

    clear_after_render = False

    if request.method == "POST":
        user_input = request.form["message"]
        lower_input = user_input.lower()

        # Keywords detection
        hadith_keywords = ["hadith", "islam", "prophet", "sunnah", "muhammad", "muslim", "deen"]
        university_keywords = ["apply", "admission", "requirements", "date", "contact", "scholarship"]

        if any(word in lower_input for word in hadith_keywords):
            # Handle HadithBot
            hadith_results = search_hadiths(user_input)
            reply = "<br>".join([f"<strong>Hadith {i+1}:</strong> {hadith}" for i, hadith in enumerate(hadith_results)])

        elif any(word in lower_input for word in university_keywords):
            # Handle UniversityBot
            if any(greet in lower_input for greet in ["hi", "hello", "hey", "salam", "how are you"]):
                reply = responses["greeting"]
            elif "apply" in lower_input:
                reply = responses["apply"]
            elif "requirement" in lower_input:
                reply = responses["requirements"]
            elif "date" in lower_input:
                reply = responses["dates"]
            elif "contact" in lower_input:
                reply = responses["contact"]
            elif "scholarship" in lower_input:
                reply = responses["scholarships"]
            elif "thanks" in lower_input or "thank you" in lower_input:
                reply = "Thanks for using the bot! "
                session["chat_history"].append(("You", user_input))
                session["chat_history"].append(("Bot", reply))
                session["clear_next"] = True
                session.modified = True
                return redirect(url_for('index'))
            else:
                reply = "Sorry, I didn’t get that. <br>Try typing:<br>- apply<br>- requirements<br>- dates<br>- contact<br>- scholarships<br>- or say hi!"

        else:
            # User message unclear
            reply = "Please specify whether you want Hadith knowledge or University admission help."

        session["chat_history"].append(("You", user_input))
        session["chat_history"].append(("Bot", reply))
        session.modified = True

    elif request.method == "GET" and session.get("clear_next"):
        clear_after_render = True

    rendered_page = render_template("index.html", chat_history=session["chat_history"])

    if clear_after_render:
        session["chat_history"] = []
        session.pop("clear_next", None)
        session.modified = True

    return rendered_page


@app.route("/clear", methods=["POST"])
def clear():
    session["chat_history"] = []
    return redirect(url_for("index"))

# -------------------------
# Run the app
# -------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
