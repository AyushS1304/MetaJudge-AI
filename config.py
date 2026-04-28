"""
config.py — Central config for Skeptical CoVe-RAG
Set your GROQ_API_KEY in a .env file or as an environment variable.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Model tiers — swap as needed
# Fast, cheap: llama-3.1-8b-instant
# Strong:      llama-3.3-70b-versatile  or  mixtral-8x7b-32768
FAST_MODEL   = "llama-3.1-8b-instant"
STRONG_MODEL = "llama-3.3-70b-versatile"

# How many atomic facts to verify per run (set lower for testing)
MAX_FACTS = 20

# Retrieval
MAX_ARXIV_RESULTS = 3
MAX_WEB_RESULTS   = 3

# CoVe
MIN_EVIDENCE_CHARS = 30   # Evidence quote must be at least this long to count as grounded