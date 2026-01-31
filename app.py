from searcher import EmbeddedSearcher, load_image_paths

from flask import Flask, request, jsonify, url_for
from werkzeug.utils import secure_filename
from markupsafe import escape
import os

ITEMS = "static/items/"
model = EmbeddedSearcher(ITEMS)

app = Flask(__name__, static_folder='static')

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


ALLOWED_EXTENSIONS = {'png', 'jpg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post("/api/upload")
def upload():
    file = request.files["item"]
    if file is None: return jsonify({ "code": 1 })
    if not allowed_file(file.filename): return jsonify({ "code": 2 })

    filepath = f"{ITEMS}{secure_filename(file.filename)}"
    if os.path.exists(filepath): return jsonify({ "code": 3 })

    file.save(filepath)
    model.update()

    return jsonify({ "code": 0, "path": filepath })

@app.post("/api/remove")
def remove():
    data = request.get_json()
    item = data.get("item")
    if item is None: return jsonify({ "code": 1 })

    filepath = f"{ITEMS}{secure_filename(item)}"
    if not os.path.exists(filepath): return jsonify({ "code": 2 })

    os.remove(filepath)
    model.update()

    return jsonify({ "code": 0 })

if __name__ == "__main__":
    app.run(port=3001)