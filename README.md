# TIRAMISO
TIRAMISO (Transformer-based Item Recognition for Actively Missing Objects) is Camp Talusi's submission
for the 2026 STEM Week AI Hackathon.

# How to Use
0. Install dependencies (i.e. `pip install -r requirements.txt`)
1. Create a `secret` file which will be used to create Flask sessions
2. Create a `credentials.json` file with the following structure:
```json
{
    "credentials": {
        "username1": "hash1", 
        "username2": "hash2",
        "username3": "hash3",
        ...
    }
}
```
> [!NOTE]
> Hashes can be generated using the following in Python:
> ```py
> from werkzeug.security import generate_password_hash
> generate_password_hash("password here")
> ```

3. Run `app.py` (i.e. `python app.py`)

# Technologies Used
- Python
- Flask
- PyTorch
- Pillow
- OpenAI CLIP

# Developers
- zowlorin (AI model development)
- xi_pec (Frontend and Backend development)