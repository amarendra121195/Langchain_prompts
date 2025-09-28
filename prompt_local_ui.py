import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# -------------------------------
# Streamlit UI Header
# -------------------------------
st.set_page_config(page_title="AI Research Summarizer", layout="centered")
st.title("AI Research Summarizer Tool")

# -------------------------------
# Research Paper Options
# -------------------------------
paper_options = [
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers",
    "GPT-3: Language Models are Few-Shot Learners",
    "Diffusion Models Beat GANs on Image Synthesis"
]

paper_input = st.selectbox("Select Research Paper", paper_options)

# -------------------------------
# Explanation Style Options
# -------------------------------
style_options = ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
style_input = st.selectbox("Select Explanation Style", style_options)

# -------------------------------
# Explanation Length Options
# -------------------------------
length_options = ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
length_input = st.selectbox("Select Explanation Length", length_options)

# -------------------------------
# Load Local Hugging Face Model
# -------------------------------
st.info("Loading local model. This may take a few seconds...")

model_name = "distilgpt2"  # small CPU-friendly model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32  # CPU-friendly
)

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    device=-1  # CPU
)

st.success("Model loaded successfully!")

# -------------------------------
# Generate Summary
# -------------------------------
if st.button("Summarize"):
    prompt = f"""
    Summarize the research paper titled "{paper_input}".
    Style: {style_input}
    Length: {length_input}
    """
    
    with st.spinner("Generating summary..."):
        generated = text_generator(prompt)
        summary_text = generated[0]["generated_text"]
        st.subheader("Generated Summary")
        st.write(summary_text)
