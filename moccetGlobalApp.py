import os
import re
import fitz  # PyMuPDF
import traceback
from datetime import datetime
from PIL import Image
import pytesseract
import tempfile
import requests
from flask import Flask, request, make_response
from dotenv import load_dotenv

import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt.oauth.oauth_settings import OAuthSettings
from slack_sdk.oauth.installation_store import InstallationStore
from slack_sdk.oauth.state_store import OAuthStateStore
from slack_sdk import WebClient

import time
import secrets
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

flask_app = Flask(__name__)
flask_app.secret_key = os.getenv("FLASK_SECRET_KEY", "super-secret-key")

# --- In-memory OAuth stores ---
class SimpleInstallationStore(InstallationStore):
    def __init__(self):
        self.installations = {}

    def save(self, installation):
        self.installations[installation.team_id] = installation
        logger.info(f"Installation saved: team_id={installation.team_id}")

    def find_installation(self, *, team_id, **kwargs):
        installation = self.installations.get(team_id)
        logger.info(f"Installation lookup: team_id={team_id}, found={installation is not None}")
        return installation

    def delete_installation(self, *, team_id, **kwargs):
        if team_id in self.installations:
            del self.installations[team_id]
            logger.info(f"Installation deleted: team_id={team_id}")

class SimpleOAuthStateStore(OAuthStateStore):
    def __init__(self):
        self.states = {}

    def issue(self) -> str:
        state = secrets.token_urlsafe(16)
        self.states[state] = time.time()
        logger.info(f"State issued: {state}")
        return state

    def consume(self, state: str) -> bool:
        if state in self.states:
            del self.states[state]
            logger.info(f"State consumed: {state}")
            return True
        logger.info(f"State not found: {state}")
        return False

installation_store = SimpleInstallationStore()
state_store = SimpleOAuthStateStore()

oauth_settings = OAuthSettings(
    client_id=os.getenv("SLACK_CLIENT_ID"),
    client_secret=os.getenv("SLACK_CLIENT_SECRET"),
    scopes=os.getenv("SLACK_SCOPES").split(","),
    installation_store=installation_store,
    state_store=state_store,
    redirect_uri=os.getenv("SLACK_REDIRECT_URI"),
)

bolt_app = App(
    signing_secret=os.getenv("SLACK_SIGNING_SECRET"),
    oauth_settings=oauth_settings,
)

handler = SlackRequestHandler(app=bolt_app)

# Gemini setup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# PDF processing
local_pdf_folder = "./PMPDFs"

def extract_text_with_ocr_from_doc(doc):
    text = ""
    for page in doc:
        page_text = page.get_text().strip()
        if not page_text:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img)
            text += ocr_text + "\n"
        else:
            text += page_text + "\n"
    return text

def preload_pdfs_from_folder(folder_path):
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            try:
                with fitz.open(file_path) as doc:
                    text = extract_text_with_ocr_from_doc(doc)
                    all_text += text + "\n"
            except Exception:
                traceback.print_exc()
    return all_text

initial_text = preload_pdfs_from_folder(local_pdf_folder)
chunks = text_splitter.split_text(initial_text)
vectorstore = FAISS.from_texts(chunks, embedding_model)

def get_advice(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        traceback.print_exc()
        return "Sorry, something went wrong."

def clean_slack_text(text):
    return re.sub(r"<@[\w]+>", "", text).strip()

# Slack WebClient
slack_bot_token = os.getenv("SLACK_BOT_TOKEN")
client = WebClient(token=slack_bot_token)

def fetch_last_messages(channel_id, limit=10):
    try:
        result = client.conversations_history(channel=channel_id, limit=limit)
        messages = [
            msg["text"]
            for msg in reversed(result["messages"])
            if "text" in msg and not msg.get("subtype")
        ]
        return messages
    except Exception as e:
        logger.error(f"Failed to fetch messages: {e}")
        return []

# -------------------------------
# summarize uploaded PDFs
# -------------------------------
def summarize_pdf_from_slack(file_id):
    try:
        file_info = client.files_info(file=file_id)
        file_url = file_info["file"]["url_private_download"]
        headers = {"Authorization": f"Bearer {slack_bot_token}"}

        response = requests.get(file_url, headers=headers)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_pdf_path = tmp_file.name

        with fitz.open(tmp_pdf_path) as doc:
            # Try normal text extraction first (no OCR)
            full_text = ""
            for page in doc:
                page_text = page.get_text().strip()
                full_text += page_text + "\n"

            # If extracted text is too short, fallback to OCR
            if len(full_text.strip()) < 100:
                full_text = ""
                for page in doc:
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img)
                    full_text += ocr_text + "\n"

            full_text = full_text[:6000]  # Limit length for prompt

        summary = get_advice(f"Summarize the following document:\n\n{full_text}")
        return summary

    except Exception as e:
        logger.error(f"Failed to summarize PDF: {e}")
        return "Sorry, I couldn't process the uploaded PDF."


    except Exception as e:
        logger.error(f"Failed to summarize PDF: {e}")
        return "Sorry, I couldn't process the uploaded PDF."

# --------------------------------
#  Message + App Mention Events
# --------------------------------
@bolt_app.event("app_mention")
def handle_mention_events(body, say):
    user = body["event"]["user"]
    channel_id = body["event"]["channel"]
    question = clean_slack_text(body["event"]["text"])

    recent_messages = fetch_last_messages(channel_id, limit=10)
    conversation_context = "\n".join(clean_slack_text(msg) for msg in recent_messages)

    relevant_docs = vectorstore.similarity_search(question, k=4)
    document_context = "\n---\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""You're an expert project advisor bot. Use both the recent Slack conversation and the project documents to help the user.

Slack conversation (last 10 messages):
{conversation_context}

Relevant project documents:
{document_context}

User's question:
{question}

Answer:"""

    reply = get_advice(prompt)
    say(f"<@{user}>, here's Moccet's advice:\n\n{reply}")

# ----------------------------
#  Handle uploaded PDF files
# ----------------------------
@bolt_app.event("message")
def handle_file_uploads(body, say, logger):
    event = body.get("event", {})
    subtype = event.get("subtype")

    if subtype == "file_share":
        file_id = event["files"][0]["id"]
        channel_id = event["channel"]
        logger.info(f"PDF file shared in channel: {channel_id}")
        summary = summarize_pdf_from_slack(file_id)
        say(channel=channel_id, text=f"Here's a summary of the uploaded document:\n\n{summary}")
    else:
        logger.info("Received a general message event.")

# -----------------------
#  Flask HTTP Routes
# -----------------------
@flask_app.before_request
def before():
    logger.info(f"Incoming request: {request.method} {request.path}")
    logger.debug(f"Headers: {request.headers}")

@flask_app.route("/slack/install", methods=["GET"])
def install():
    return handler.handle(request)

@flask_app.route("/slack/oauth_redirect", methods=["GET"])
def oauth_callback():
    logger.info(f"OAuth callback query params: {request.args}")
    try:
        return handler.handle(request)
    except Exception as e:
        logger.error(f"OAuth callback failed: {str(e)}")
        return make_response("OAuth failed", 500)

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    try:
        data = request.get_json()
        logger.info(f"Received data: {data}")

        if data and data.get("type") == "url_verification":
            logger.info("Responding to URL verification challenge.")
            return make_response(data["challenge"], 200, {"Content-Type": "text/plain"})
    except Exception as e:
        logger.error(f"Error processing request: {e}")

    return handler.handle(request)

@flask_app.route("/healthcheck")
def healthcheck():
    return "OK",

if __name__ == "__main__":
    flask_app.run(port=3000, debug=True)
