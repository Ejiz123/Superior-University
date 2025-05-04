from flask import Flask, render_template, request, session, redirect, url_for
from flask_session import Session

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

responses = {
    "greeting": "Hi there! Welcome to your University Admission Assistant!",
    "apply": "To apply for admission, visit: https://youruniversity.edu/apply",
    "requirements": "Requirements:<br>Undergraduate: 50% in Intermediate.<br>Graduate: 2.5 CGPA & test/interview.",
    "dates": "Important Dates:<br>Deadline: 30th June<br>Entry Test: 10th July<br>Merit List: 15th July<br>Classes: 1st August",
    "contact": "Contact Us:<br>Phone: +92-300-0000000<br>Email: admissions@youruniversity.edu",
    "scholarships": "Scholarships:<br>Merit-based, Need-based, Talent-based.<br>Visit: https://youruniversity.edu/scholarships"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []

    clear_after_render = False  # flag to clear after rendering

    if request.method == "POST":
        user_input = request.form["message"]
        lower_input = user_input.lower()

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

        session["chat_history"].append(("You", user_input))
        session["chat_history"].append(("Bot", reply))
        session.modified = True

    elif request.method == "GET" and session.get("clear_next"):
        clear_after_render = True  # set the flag but don’t clear yet

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

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
