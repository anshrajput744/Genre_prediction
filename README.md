

Genre Prediction and Book Recommendation System

Project Overview
This project aims to enhance book discovery by predicting a book’s genre based on its metadata (e.g., title, description) and providing personalized book recommendations. The system uses machine learning and natural language processing (NLP) to classify genres and a recommendation engine to suggest similar books to users based on their preferences or selected titles.

Key Components
1. Genre Prediction Model
   - Predicts the genre of a book using textual features such as the book's title, summary, or description.
   - Utilizes NLP techniques to extract meaningful features and classify text into predefined genres (e.g., Fiction, Romance, Mystery, Sci-Fi).

2. Recommendation System
   - Suggests books similar to a selected book or a user’s preferences.
   - Can be content-based (based on book attributes and descriptions) or collaborative (based on user behavior and preferences).

Dataset
- Source: Public book datasets from sources like Goodreads, Google Books API, or Kaggle datasets.
- Features: Book title, author, description/summary, genre labels, user ratings.
- **Size:** ~50,000+ books (varies by dataset)

### **Technologies Used**
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK/spaCy, TensorFlow/Keras, LightFM, Surprise  
- **Tools:** Jupyter Notebook, Tkinter (for GUI integration), REST APIs (Google Books, Wikipedia)

### **Modeling Techniques**
- Genre Classification: 
  - NLP pipeline with tokenization, TF-IDF vectorization, and classification using models like Logistic Regression, Naive Bayes, or BERT.
- Recommendation:
  - Content-Based Filtering: Cosine similarity on TF-IDF vectors of descriptions.
  - Collaborative Filtering: Matrix factorization techniques like SVD or neural embeddings.

Evaluation Metrics
- Genre Prediction: Accuracy, Precision, Recall, F1-score  
- Recommendations: Precision@K, Recall@K, Mean Average Precision (MAP), and qualitative analysis via user testing

User Interface
- A simple desktop GUI built with Tkinter allows users to:
  - Enter a book title and get a summary and genre prediction.
  - Receive 5–10 similar book recommendations with links to summaries and purchase pages.

Results
- Genre Prediction Accuracy: ~85–90% on multi-class genre classification
- Recommendation Quality: High user relevance with precision@5 > 0.75 (depending on dataset)

Potential Improvements
- Incorporating deep learning models like BERT for better understanding of book descriptions
- Using hybrid recommendation models combining content and collaborative filtering
- Enhancing the UI with search filters and personalization features
