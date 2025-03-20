import streamlit as st
import requests

# Streamlit UI
st.title("Expense Entity Recognition")

# User input
user_input = st.text_area("Enter an expense description:", "I spent $50 at Walmart on 2024-06-10 using my Credit Card.")

if st.button("Extract Entities"):
    response = requests.post("http://127.0.0.1:8000/extract-expenses", json={"text": user_input})
    
    if response.status_code == 200:
        entities = response.json().get("extracted_entities", [])
        if entities:
            st.success("Extracted Entities:")
            for entity in entities:
                st.write(f"**{entity['label']}**: {entity['text']}")
        else:
            st.warning("No entities detected.")
    else:
        st.error("Error connecting to API.")
