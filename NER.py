import streamlit as st
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    pipeline,
    BertForTokenClassification
)

# Load models
ner_model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer_ner = BertTokenizer.from_pretrained(ner_model_name)
model_ner = BertForTokenClassification.from_pretrained(ner_model_name)

masked_model_name = "bert-base-uncased"
tokenizer_masked = BertTokenizer.from_pretrained(masked_model_name)
model_masked = BertForMaskedLM.from_pretrained(masked_model_name)

# Set up pipelines
ner_pipeline = pipeline("ner", model=model_ner, tokenizer=tokenizer_ner)
masked_pipeline = pipeline("fill-mask", model=model_masked, tokenizer=tokenizer_masked)

# Streamlit App
st.title("NER and Sentence Completion with BERT")

# Option to select between NER and Sentence Completion
task = st.sidebar.selectbox("Choose a task", ["Named Entity Recognition", "Sentence Completion"])

# Named Entity Recognition (NER) Task
if task == "Named Entity Recognition":
    st.header("Named Entity Recognition (NER)")
    st.write("Enter text and the app will identify named entities such as persons, organizations, locations, etc.")
    
    user_input = st.text_area("Enter your text here:")
    
    if st.button("Identify Entities"):
        if user_input:
            entities = ner_pipeline(user_input)
            if entities:
                st.write("### Identified Entities:")
                for entity in entities:
                    st.write(f"**Entity**: `{entity['word']}`, **Type**: `{entity['entity']}`, **Score**: `{entity['score']:.4f}`")
            else:
                st.write("No entities identified.")
        else:
            st.warning("Please enter some text for entity recognition.")

# Sentence Completion Task
elif task == "Sentence Completion":
    st.header("Sentence Completion with BERT")
    st.write("Enter a sentence with a missing word represented by `[MASK]` and the model will suggest completions.")
    
    user_input_mask = st.text_area("Enter your sentence with [MASK]:", "The capital of France is [MASK].")
    
    if st.button("Complete Sentence"):
        if "[MASK]" in user_input_mask:
            completions = masked_pipeline(user_input_mask)
            st.write("### Suggested Completions:")
            for completion in completions:
                st.write(f"**Suggestion**: `{completion['sequence']}`, **Score**: `{completion['score']:.4f}`")
        else:
            st.warning("Please make sure your sentence contains a `[MASK]` token.")

# Footer with additional info
st.markdown("---")
st.write("### About This App")
st.write(
    "This application uses BERT models for two main tasks: Named Entity Recognition (NER) "
    "and Sentence Completion. The NER model identifies entities in the text, while the "
    "Sentence Completion model predicts missing words in a sentence."
)
