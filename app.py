from searcher import EmbeddedSearcher, load_image_paths

from flask import Flask, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from markupsafe import escape
import os
import json

ITEMS = "static/items/"
model = EmbeddedSearcher(ITEMS)

app = Flask(__name__, static_folder='static')
with open('secret') as f: app.secret_key = f.read()
with open('credentials.json') as f: credentials = json.load(f)["credentials"]

@app.get("/api/search")
def search():
    query = request.args.get("query", type=str)
    start = request.args.get("start", 0, type=int)
    count = request.args.get("count", 5, type=int)

    if query is None: return jsonify({ "code": 1 })

    items = model.query(query, start, count)
    formatted = [{ "path": path, "confidence": confidence } for path, confidence in items]
    return jsonify({ "code": 0, "items": formatted })


@app.get("/api/list")
def list():
    files = load_image_paths(ITEMS)
    urls = [path.replace(os.sep, "/") for path in files]
    return jsonify({ "code": 0, "items": urls })


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post("/api/upload")
def upload():
    if "user" not in session: return jsonify({ "code": 1 })

    file = request.files["item"]
    if file is None: return jsonify({ "code": 2 })
    if not allowed_file(file.filename): return jsonify({ "code": 3 })

    filepath = f"{ITEMS}{secure_filename(file.filename)}"
    if os.path.exists(filepath): return jsonify({ "code": 4 })

    file.save(filepath)
    model.update()

    return jsonify({ "code": 0, "path": filepath })

@app.post("/api/remove")
def remove():
    if "user" not in session: return jsonify({ "code": 1 })

    data = request.get_json()
    item = data.get("item")
    if item is None: return jsonify({ "code": 2 })

    filepath = f"{ITEMS}{secure_filename(item)}"
    if not os.path.exists(filepath): return jsonify({ "code": 3 })

    os.remove(filepath)
    model.update()

    return jsonify({ "code": 0 })

@app.post("/api/login")
def login():
    if "user" in session: return jsonify({ "code": 1 })

    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    
    if username in credentials and check_password_hash(credentials[username], password):
        session["user"] = username
        return jsonify({ "code": 0 })
    else: return jsonify({ "code": 1 })

@app.post("/api/logout")
def logout():
    if "user" not in session: return jsonify({ "code": 1 })

    session.clear()
    return jsonify({ "code": 0 })

@app.get("/api/validate")
def validate():
    return jsonify({ "code": 0 if "user" in session else 1 })

if __name__ == "__main__":
    app.run(port=3001)