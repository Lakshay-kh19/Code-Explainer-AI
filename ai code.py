import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

@st.cache_resource
def initialize_model():
    model_id = "EleutherAI/gpt-neo-125M"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return tokenizer, model

def explain_python_code(code: str, tokenizer, model) -> str:
    prompt = (
        "You are an expert Python tutor. Please explain the following code clearly and simply:\n\n"
        f"{code}\n\nExplanation:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        top_p=0.95,
        temperature=0.6,
    )
    explanation = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return explanation.strip()

def launch_app():
    st.title("🧠 CodeComprehend: Python Code Explainer")
    tokenizer, model = initialize_model()

    user_input = st.text_area("✏️ Enter Python code below:", height=200)
    if st.button("Explain Code") and user_input.strip():
        with st.spinner("Generating smart explanation..."):
            result = explain_python_code(user_input, tokenizer, model)
        st.markdown("### 📘 Explanation")
        st.write(result)

if __name__ == "__main__":
    launch_app()
