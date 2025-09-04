import streamlit as st

# ------------------------------
# UI CONFIG
# ------------------------------
st.set_page_config(page_title="Open LLM Chatbot", page_icon="🤖", layout="wide")

# Define the pages
main_page = st.Page("main_page.py", title="Inicio", icon="🏠")

# Set up navigation
pg = st.navigation([main_page])

# Run the selected page
pg.run()

