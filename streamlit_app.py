import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_model_and_tokenizer(repo_id):
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
    return model, tokenizer

repo_id = "kodamkarthik281/t5-cnn-summary-karthi"
model, tokenizer = load_model_and_tokenizer(repo_id)

def generate_summary(text, model, tokenizer, max_input_len=512, max_output_len=150, num_beams=4):
    input_ids = tokenizer.encode(
        "summarize: " + text,
        return_tensors="pt",
        max_length=max_input_len,
        truncation=True
    )
    input_ids = input_ids.to(model.device)
    output_ids = model.generate(
        input_ids,
        max_length=max_output_len,
        num_beams=num_beams,
        early_stopping=True
    )
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary

st.set_page_config(page_title="Abstractive Text Summarizer", layout="wide")
st.title("Abstractive Text Summarizer")

st.markdown("""
Paste your paragraph below and get the abstractive summary using the fine-tuned model.
""")

user_input = st.text_area("Paste your paragraph:", height=150)

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Summarize"):
    if user_input.strip():
        with st.spinner("Generating summary..."):
            
            st.markdown("**Note :** This app may take 2–3 minutes to generate a summary after clicking the button.", unsafe_allow_html=True)
            st.markdown("""**Why is it slow? :** The model is a fine-tuned Transformer (T5) loaded from Hugging Face. Due to limited compute resources on Hugging
            Face Spaces (CPU-only and shared infrastructure), initial inference can take some time. Please be patient.""", unsafe_allow_html=True)

            abs_summary = generate_summary(user_input, model, tokenizer)
        st.subheader("Abstractive Summary")
        st.success(abs_summary)

        st.session_state.history.append({
            "input": user_input,
            "abstractive": abs_summary
        })
    else:
        st.warning("Please enter a paragraph to summarize.")

if st.session_state.history:
    st.markdown("---")
    st.subheader("Summary History")
    for i, item in enumerate(reversed(st.session_state.history), 1):
        with st.expander(f"Example #{i}"):
            st.markdown("**Original Text:**")
            st.write(item["input"])
            st.markdown("**Abstractive Summary:**")
            st.success(item["abstractive"])
