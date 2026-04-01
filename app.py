from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from youtube_comment_downloader import YoutubeCommentDownloader
from wordcloud import WordCloud
import base64
import io
import itertools

app = Flask(__name__)

# ── 1. LOAD MODEL & VECTORIZER ────────────────────────────────
print("Loading model...")
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("✅ Model loaded!")

# ── 2. SAME CLEANING FUNCTION AS TRAINING ────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
keep_words = {'not', 'no', 'nor', 'but', 'very', 'too'}
stop_words = stop_words - keep_words

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# ── 3. PREDICT FUNCTION ───────────────────────────────────────
def predict_sentiment(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    confidence = round(max(probability) * 100, 2)
    label = 'Positive' if prediction == 1 else 'Negative'
    return label, confidence

def generate_wordcloud(text):
    try:
        # Only remove URLs and special characters, nothing else
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove numbers, symbols
        text = re.sub(r'\b\w{1,2}\b', '', text)   # remove 1-2 letter words only

        wc = WordCloud(
            width=800, height=500,
            background_color='#0f1117',
            colormap='cool',
            max_words=80,
            prefer_horizontal=0.9,
            stopwords=None
        ).generate(text)
        buf = io.BytesIO()
        wc.to_image().save(buf, format='PNG')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        return f'data:image/png;base64,{img_b64}'
    except:
        return None
# ── 4. ROUTES ─────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')

# Bulk text analysis (Instagram, Twitter, any platform)
@app.route('/analyze_bulk', methods=['POST'])
def analyze_bulk():
    raw_text = request.form.get('comments', '')
    comments = [line.strip() for line in raw_text.strip().split('\n') if line.strip()]

    if not comments:
        return jsonify({'error': 'No comments provided'})

    results = []
    positive_count = 0
    negative_count = 0

    for comment in comments:
        label, confidence = predict_sentiment(comment)
        results.append({
            'comment': comment,
            'sentiment': label,
            'confidence': confidence
        })
        if label == 'Positive':
            positive_count += 1
        else:
            negative_count += 1

    total = len(results)
    positive_pct = round((positive_count / total) * 100, 1)
    negative_pct = round((negative_count / total) * 100, 1)

    # Generate word cloud
    try:
        all_text = ' '.join([r['comment'] for r in results])
        wc_image = generate_wordcloud(all_text)
    except:
        wc_image = None

    return jsonify({
        'results': results,
        'wordcloud': wc_image,
        'summary': {
            'total': total,
            'positive': positive_count,
            'negative': negative_count,
            'positive_pct': positive_pct,
            'negative_pct': negative_pct
        }
    })

# YouTube URL analysis
@app.route('/analyze_youtube', methods=['POST'])
def analyze_youtube():
    url = request.form.get('youtube_url', '')
    if not url:
        return jsonify({'error': 'No URL provided'})

    try:
        video_id = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', url)
        if not video_id:
            return jsonify({'error': 'Invalid YouTube URL'})
        video_id = video_id.group(1)

        import requests as req_lib
        api_url = "https://youtube-media-downloader.p.rapidapi.com/v2/video/comments"
        headers = {
            "x-rapidapi-host": "youtube-media-downloader.p.rapidapi.com",
            "x-rapidapi-key": "36c443cff2msh178c1f72964eed0p1a6d60jsnd8ab4f6469f5"
        }

        all_comments = []
        next_token = None
        max_pages = 5  # fetches up to 100 comments (20 x 5 pages)

        for page in range(max_pages):
            params = { "videoId": video_id, "sortBy": "top" }
            if next_token:
                params["nextToken"] = next_token

            response = req_lib.get(api_url, headers=headers, params=params)
            data = response.json()

            if 'items' not in data:
                break

            all_comments.extend(data['items'])
            next_token = data.get('nextToken')

            if not next_token:
                break

        if not all_comments:
            return jsonify({'error': 'No comments found or invalid URL'})

        results = []
        positive_count = 0
        negative_count = 0

        for item in all_comments:
            comment = item.get('contentText', '')
            if not comment.strip():
                continue
            label, confidence = predict_sentiment(comment)
            results.append({
                'comment': comment[:200],
                'sentiment': label,
                'confidence': confidence
            })
            if label == 'Positive':
                positive_count += 1
            else:
                negative_count += 1

        total = len(results)
        if total == 0:
            return jsonify({'error': 'No valid comments found'})

        positive_pct = round((positive_count / total) * 100, 1)
        negative_pct = round((negative_count / total) * 100, 1)
        try:
            all_text = ' '.join([r['comment'] for r in results[:50]])
            wc_image = generate_wordcloud(all_text)
        except:
            wc_image = None

        return jsonify({
            'results': results,
            'wordcloud': wc_image,
            'summary': {
                'total': total,
                'positive': positive_count,
                'negative': negative_count,
                'positive_pct': positive_pct,
                'negative_pct': negative_pct
            }
        })

    except Exception as e:
        return jsonify({'error': 'YouTube fetching is only available in the local version. Please use the Paste Comments tab for the live demo, or run the app locally for YouTube analysis.'})


if __name__ == '__main__':
    app.run(debug=True)
