from flask import Flask, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from scipy.special import softmax
import logging

# Initialize Flask app
app = Flask(__name__)

# Load NLTK's VADER
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Hugging Face API key (ensure this is set in your environment variables)
HUGGING_FACE_API_KEY = "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"

def analyze_sentiment(text):
    # VADER sentiment analysis
    vader_result = sia.polarity_scores(text)

    # RoBERTa sentiment analysis via Hugging Face Inference API
    api_url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
    payload = {"inputs": text}

    response = requests.post(api_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        api_result = response.json()

        # Assuming the API returns logits that we need to softmax
        try:
            scores = softmax([api_result[0]['score'] for result in api_result])
            roberta_result = {
                'roberta_neg': scores[0],
                'roberta_neu': scores[1],
                'roberta_pos': scores[2]
            }
        except (IndexError, KeyError) as e:
            logging.error(f"Error processing API response: {e}")
            roberta_result = {
                'roberta_neg': None,
                'roberta_neu': None,
                'roberta_pos': None
            }
    else:
        logging.error(f"API request failed with status code {response.status_code}")
        roberta_result = {
            'roberta_neg': None,
            'roberta_neu': None,
            'roberta_pos': None
        }

    return {**vader_result, **roberta_result}

def sentiment_to_stars(sentiment_score):
    thresholds = [0.2, 0.4, 0.6, 0.8]
    if sentiment_score is None:
        return 0
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
async def analyze():
    data = await request.json
    text = data['text']
    sentiment_scores = analyze_sentiment(text)
    star_rating = sentiment_to_stars(sentiment_scores.get('roberta_pos'))
    response = {
        'sentiment_scores': sentiment_scores,
        'star_rating': star_rating
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
