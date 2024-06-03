from flask import Flask, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from scipy.special import softmax

# Initialize Flask app
app = Flask(__name__)

# Load NLTK's VADER
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    # VADER sentiment analysis
    vader_result = sia.polarity_scores(text)

    # RoBERTa sentiment analysis via Hugging Face Inference API
    api_url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
    headers = {"Authorization": "Bearer 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"}  # Replace YOUR_HUGGING_FACE_API_KEY with your actual key
    payload = {"inputs": text}

    response = requests.post(api_url, headers=headers, json=payload)
    api_result = response.json()

    # Ensure the structure matches expected keys
    labels = ['roberta_neg', 'roberta_neu', 'roberta_pos']
    scores = softmax([result['score'] for result in api_result[0]])
    roberta_result = dict(zip(labels, scores))

    return {**vader_result, **roberta_result}

def sentiment_to_stars(sentiment_score):
    thresholds = [0.2, 0.4, 0.6, 0.8]
    if sentiment_score <= thresholds[0]:
        return 1
    elif sentiment_score <= thresholds[1]:
        return 2
    elif sentiment_score <= thresholds[2]:
        return 3
    elif sentiment_score <= thresholds[3]:
        return 4
    else:
        return 5

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data['text']
    sentiment_scores = analyze_sentiment(text)
    star_rating = sentiment_to_stars(sentiment_scores['roberta_pos'])
    response = {
        'sentiment_scores': sentiment_scores,
        'star_rating': star_rating
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
