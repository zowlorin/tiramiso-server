from searcher import EmbeddedSearcher, load_image_paths

from flask import Flask, request, jsonify, url_for
from markupsafe import escape
import os

SAMPLES_PATH = "static/items/"
model = EmbeddedSearcher(SAMPLES_PATH)

app = Flask(__name__, static_folder='static')

@app.get("/api/search")
def search():
    query = request.args.get("query", type=str)
    start = request.args.get("start", 0, type=int)
    count = request.args.get("count", 5, type=int)

    if query is None: return jsonify({ "code": 1 })

    items = model.query(query, start, count)
    formatted = [{ "path": path, "confidence": confidence } for path, confidence in items]
    return jsonify({ "items": formatted })


@app.get("/api/list")
def list():
    files = load_image_paths(SAMPLES_PATH)
    urls = [path.replace(os.sep, "/") for path in files]
    return jsonify({ "items": urls })

if __name__ == "__main__":
    app.run(port=3001)