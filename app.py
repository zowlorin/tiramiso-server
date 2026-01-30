from searcher import EmbeddedSearcher

from flask import Flask
from markupsafe import escape

app = Flask(__name__)

@app.route("/")

def root():
    return "<p>Hello, World!</p>"

# SAMPLES_PATH = "samples/"

# if __name__ == "__main__":
#     model = EmbeddedSearcher(SAMPLES_PATH)

#     while True:
#         query = input("Query: ")

#         print(model.query(query,5))