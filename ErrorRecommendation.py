import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to calculate cosine similarity
def calculate_similarity(input_text, error_messages):
    vectorizer = TfidfVectorizer()
    all_texts = [input_text] + error_messages
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    input_vector = tfidf_matrix[0] # type: ignore
    error_vectors = tfidf_matrix[1:] # type: ignore
    cosine_similarities = cosine_similarity(input_vector, error_vectors).flatten()
    
    return cosine_similarities

# Streamlit app
def main():
    st.title("Error Message Similarity Checker")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        
        # Check if the CSV has the required columns
        if 'Error' not in df.columns or 'Solution' not in df.columns:
            st.error("The CSV file must contain 'Error' and 'Solution' columns.")
            return

        # Input text box
        input_text = st.text_area("Enter your error message here:", height=100)

        if st.button("Compare"):
            if input_text:
                try:
                    # Calculate similarities
                    similarities = calculate_similarity(input_text, df['Error'].tolist())
                    
                    # Add similarities to the DataFrame
                    df['Similarity'] = similarities
                    
                    # Sort by similarity in descending order
                    df_sorted = df.sort_values('Similarity', ascending=False)
                    
                    # Display the full table
                    st.subheader("All Errors with Similarities")
                    st.dataframe(df_sorted)
                    
                    # Display top 5 matching errors
                    st.subheader("Top 5 Matching Errors")
                    top_5 = df_sorted.head()
                    st.table(top_5[['Error', 'Solution', 'Similarity']])
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("Please enter an error message to compare.")
    else:
        st.info("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
