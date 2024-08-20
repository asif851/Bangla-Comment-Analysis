from flask import Flask, request, render_template
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Load the model and vectorizer
model = joblib.load('comment_analysis_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize Flask application
app = Flask(__name__)
    
def preprocess_text(text):
    
    # Remove non-alphabetic characters
    text = "".join([char for char in text if char.isalpha() or char.isspace()])

    # Remove extra whitespace
    text = text.strip()

    # Load Bengali stopwords
    stop_words = stopwords.words("bengali")
    
    # Tokenize the text and remove stop words
    text = " ".join([word for word in word_tokenize(text) if word not in stop_words])
    
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        comment = request.form['comment']
        processed_comment = preprocess_text(comment)
        features = vectorizer.transform([processed_comment])
        prediction = model.predict(features)
        return render_template('index.html', prediction=prediction[0], comment=comment)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True,port= 3000)
