from flask import request, jsonify
from app import app

@app.route("/")
def index():
    return "ConCore is running!"

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    return jsonify({"message": f"{file.filename} received"}), 200
