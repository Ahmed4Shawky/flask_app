from flask import Flask, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import aiohttp
import asyncio
from scipy.special import softmax
from asgiref.wsgi import WsgiToAsgi

# Initialize Flask app
app = Flask(__name__)

# Load NLTK's VADER
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

async def fetch_roberta_sentiment(text, session):
    api_url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
    headers = {"Authorization": "Bearer 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"}  
    payload = {"inputs": text}
    async with session.post(api_url, headers=headers, json=payload) as response:
        return await response.json()

async def analyze_sentiment(text):
    # VADER sentiment analysis
    vader_result = sia.polarity_scores(text)

    async with aiohttp.ClientSession() as session:
        api_result = await fetch_roberta_sentiment(text, session)

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

@app.route('/')
def home():
    return "Sentiment Analysis API is running."

@app.route('/analyze', methods=['POST'])
async def analyze():
    data = request.json
    text = data['text']
    sentiment_scores = await analyze_sentiment(text)
    star_rating = sentiment_to_stars(sentiment_scores['roberta_pos'])
    response = {
        'sentiment_scores': sentiment_scores,
        'star_rating': star_rating
    }
    return jsonify(response)

# Wrap the Flask app with ASGI
asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(asgi_app, host='0.0.0.0', port=5000)
