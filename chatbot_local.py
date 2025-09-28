import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

st.set_page_config(page_title="Free Local Chatbot", layout="centered")
st.title("Free Local Chatbot (CPU Only)")

# -------------------------------
# Load Model
# -------------------------------
st.info("Loading model. This may take a few seconds...")

MODEL_NAME = "distilgpt2"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model **without device_map**
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Text-generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    device=-1  # CPU
)

st.success("Model loaded successfully!")

# -------------------------------
# Chat History
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# User Input
# -------------------------------
user_input = st.text_input("You:", key="input_text")

if st.button("Send"):
    if user_input:
        # Append user message
        st.session_state.history.append(f"You: {user_input}")

        # Generate response
        prompt = "\n".join(st.session_state.history) + "\nAI:"
        with st.spinner("Generating response..."):
            outputs = generator(prompt)
            ai_reply = outputs[0]["generated_text"].split("\nAI:")[-1].strip()

        # Append AI response
        st.session_state.history.append(f"AI: {ai_reply}")

# -------------------------------
# Display Chat History
# -------------------------------
if st.session_state.history:
    st.markdown("### Chat History")
    for msg in st.session_state.history:
        st.write(msg)
