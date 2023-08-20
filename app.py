import os
from flask import Flask, jsonify, request

import requests

app = Flask(__name__)

@app.route("/")
def hello():
  return "Server is running"

