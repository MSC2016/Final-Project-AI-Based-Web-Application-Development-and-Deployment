''' Executing this function initiates the emotion detection.
    It receives user input from the frontend, processes it
    using the emotion alalysis module, and returns its results.
'''
from flask import Flask, render_template, request
from EmotionDetection.emotion_detection import emotion_detector

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def render_index_page():
    '''
    Renders the main HTML page.
    Returns:
        HTML page (index.html)
    '''
    return render_template('index.html')

@app.route('/emotionDetector')
def emotion():
    '''
    Handles emotion analysis requests.
    Expects:
        Query parameter 'textToAnalyze'
    Returns:
        HTML formatted string with emotion scores and dominant emotion
    '''

    try:
        text_to_analyze = request.args.get('textToAnalyze')

        if not text_to_analyze:
            return 'Invalid input! Try again.'

        result = emotion_detector(text_to_analyze)
        print(result)

        # Handle invalid or malformed responses
        if not isinstance(result, dict) or 'dominant_emotion' not in result:
            return "Invalid text! Please try again."

        if result['dominant_emotion'] is None:
            return "Invalid text! Please try again."

        response = (
            f"<div style='font-family: monospace;'>"
            f"Anger.....: {result['anger']} <br>"
            f"Disgust...: {result['disgust']} <br>"
            f"Fear......: {result['fear']} <br>"
            f"Joy.......: {result['joy']} <br>"
            f"Sadness...: {result['sadness']} <br><br>"
            f"<b>Dominant Emotion is {result['dominant_emotion']}</b>"
            "</div>"
        )

        return response

    except (TypeError, ValueError, TimeoutError, RuntimeError) as e:
        return f'An error occurred: {str(e)}'

if __name__ == '__main__':
    # Run Flask server
    app.run(debug = True, host = '0.0.0.0', port = 5000)
