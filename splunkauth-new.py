"""
Splunk Authentication Module
Handles Splunk credentials loading from environment variables.
Credentials are optional — missing values result in empty strings (no exceptions).
"""

import os
from dotenv import load_dotenv

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path)

SPLUNK_HOST = os.getenv("SPLUNK_HOST", "192.168.15.110")
SPLUNK_PORT = os.getenv("SPLUNK_PORT", "8089")
SPLUNK_USER = os.getenv("SPLUNK_USER", "")
SPLUNK_PASSWORD = os.getenv("SPLUNK_PASSWORD", "")
