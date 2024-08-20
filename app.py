from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input text from the form
        user_input = request.form['user_input']
        
        # Transform the input text using the TF-IDF vectorizer
        input_tfidf = vectorizer.transform([user_input])
        
        # Predict the sentiment using the trained model
        prediction = model.predict(input_tfidf)[0]
        
        # Render the result on a new page or the same page
        return render_template('index.html', prediction=prediction, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)

