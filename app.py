from flask import Flask, render_template, request, jsonify
import openai
import anthropic
import google.generativeai as genai

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    model = data.get("model")
    api_key = data.get("apiKey")
    user_input = data.get("message")

    try:
        if model == "openai":
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": user_input}]
            )
            reply = response.choices[0].message["content"]

        elif model == "claude":
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=500,
                messages=[{"role": "user", "content": user_input}]
            )
            reply = response.content[0].text

        elif model == "gemini":
            genai.configure(api_key=api_key)
            model_client = genai.GenerativeModel("gemini-2.5-flash")
            response = model_client.generate_content(user_input)
            reply = response.text

        else:
            reply = "Invalid model selected."

    except Exception as e:
        reply = f"Error: {str(e)}"

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)
