#!/usr/bin/env python3
"""
Script to record interactive terminal demo cast (demo.cast) and render GIF (demo.gif)
using asciinema and agg.
"""

import json
import os
import pty
import sys
import time
import urllib.request
import subprocess
from pathlib import Path

PROMPT = "sayak@macbook ambiguard % "

def type_text(text: str, delay: float = 0.03):
    """Simulate human typing char by char."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")
    sys.stdout.flush()
    time.sleep(0.3)

def run_curl_turn1() -> tuple[str, str]:
    """Execute real turn 1 HTTP call."""
    url = "http://localhost:8000/v1/chat"
    payload = json.dumps({"question": "What are the side effects?"}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        thread_id = data.get("thread_id", "default_thread")
        
        # Use jq to format output with colors if available
        formatted_json = subprocess.check_output(
            ["jq", "."], input=json.dumps(data).encode("utf-8")
        ).decode("utf-8")
        return thread_id, formatted_json

def run_curl_turn2(thread_id: str) -> str:
    """Execute real turn 2 resume HTTP call."""
    url = "http://localhost:8000/v1/chat/resume"
    payload = json.dumps({"thread_id": thread_id, "reply": "Medication B"}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        formatted_json = subprocess.check_output(
            ["jq", "."], input=json.dumps(data).encode("utf-8")
        ).decode("utf-8")
        return formatted_json

def main():
    # 1. Clear terminal
    sys.stdout.write("\033[H\033[2J")
    sys.stdout.flush()
    time.sleep(0.5)

    # 2. Command 1: turn 1
    sys.stdout.write(PROMPT)
    sys.stdout.flush()
    cmd1 = 'curl -s localhost:8000/v1/chat -H \'Content-Type: application/json\' \\\n  -d \'{"question":"What are the side effects?"}\' | jq'
    type_text(cmd1, delay=0.02)
    
    thread_id, json1 = run_curl_turn1()
    sys.stdout.write(json1)
    sys.stdout.flush()
    time.sleep(1.5)

    # 3. Command 2: turn 2 resume
    sys.stdout.write(PROMPT)
    sys.stdout.flush()
    cmd2 = f'curl -s localhost:8000/v1/chat/resume -H \'Content-Type: application/json\' \\\n  -d \'{{"thread_id":"{thread_id}","reply":"Medication B"}}\' | jq'
    type_text(cmd2, delay=0.02)

    json2 = run_curl_turn2(thread_id)
    sys.stdout.write(json2)
    sys.stdout.flush()
    time.sleep(2.0)

if __name__ == "__main__":
    main()
