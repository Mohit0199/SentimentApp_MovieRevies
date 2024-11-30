import streamlit as st
from transformers import pipeline

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_model"
classifier = pipeline(task='sentiment-analysis', model=model_path, tokenizer=model_path)

# Streamlit app interface
st.title("Sentiment Analysis App")
st.write("Analyze the sentiment of movie reviews!")

# Text input for user review
user_input = st.text_area("Enter your movie review:", "")

# Predict sentiment when the button is clicked
if st.button("Analyze Sentiment"):
    if user_input:
        prediction = classifier(user_input)[0]
        sentiment = prediction['label']
        confidence = round(prediction['score'] * 100, 2)

        # Display result
        st.subheader("Prediction:")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence}%")
    else:
        st.warning("Please enter a review to analyze.")

