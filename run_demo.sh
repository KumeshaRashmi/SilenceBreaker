#!/usr/bin/env bash
set -e
python -m src.rag.build_index
streamlit run app/streamlit_app.py
