import unittest
from unittest.mock import patch, Mock
import requests
import sys
import os
from requests.models import Response
from requests.exceptions import HTTPError

# allow importing from parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from EmotionDetection.emotion_detection import emotion_detector, format_response

# ----------------------------
# Helper function to create mock Response
# ----------------------------
def make_mock_response(json_data=None, status_code=200, raise_for_status=None):
    mock_resp = Mock(spec=requests.models.Response)
    mock_resp.json = Mock(return_value=json_data)
    mock_resp.status_code = status_code
    if raise_for_status:
        mock_resp.raise_for_status.side_effect = raise_for_status
    else:
        mock_resp.raise_for_status = Mock()
    # support context manager (with statement)
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = Mock()
    return mock_resp

class TestEmotionDetector(unittest.TestCase):

    # ----------------------------
    # Valid input, normal case
    # ----------------------------
    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_emotion_detector_valid(self, mock_post):
        mock_json = {
            'emotionPredictions': [{
                'emotion': {'anger': 0.1, 'disgust': 0.05, 'fear': 0.05, 'joy': 0.7, 'sadness': 0.1}
            }]
        }
        mock_post.return_value = make_mock_response(json_data=mock_json)
        result = emotion_detector('I am happy')
        self.assertEqual(result['dominant_emotion'], 'joy')
        self.assertAlmostEqual(result['joy'], 0.7)
        for key in ['anger','disgust','fear','sadness']:
            self.assertIn(key, result)
            self.assertIsInstance(result[key], float)

    # ----------------------------
    # Input type error
    # ----------------------------
    def test_emotion_detector_type_error(self):
        with self.assertRaises(TypeError):
            emotion_detector(123)

    # ----------------------------
    # Input empty string
    # ----------------------------
    def test_emotion_detector_empty_string(self):
        with self.assertRaises(ValueError):
            emotion_detector('   ')

    # ----------------------------
    # Timeout
    # ----------------------------
    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_emotion_detector_timeout(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout
        with self.assertRaises(TimeoutError):
            emotion_detector('Hello world')

    # ----------------------------
    # RequestException
    # ----------------------------
    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_emotion_detector_request_exception(self, mock_post):
        mock_post.side_effect = requests.exceptions.RequestException('fail')
        with self.assertRaises(RuntimeError):
            emotion_detector('fail')

    # ----------------------------
    # format_response KeyError
    # ----------------------------
    def test_format_response_key_error(self):
        mock_json = {'emotionPredictions':[{}]}  # missing 'emotion'
        mock_resp = make_mock_response(json_data=mock_json)
        with self.assertRaises(ValueError):
            format_response(mock_resp)

    # ----------------------------
    # format_response IndexError
    # ----------------------------
    def test_format_response_index_error(self):
        mock_json = {'emotionPredictions': []}
        mock_resp = make_mock_response(json_data=mock_json)
        with self.assertRaises(ValueError):
            format_response(mock_resp)

    # ----------------------------
    # format_response invalid JSON
    # ----------------------------
    def test_format_response_invalid_json(self):
        mock_resp = make_mock_response()
        mock_resp.json.side_effect = ValueError('Invalid JSON')
        with self.assertRaises(ValueError):
            format_response(mock_resp)

    # ----------------------------
    # format_response unexpected error
    # ----------------------------
    def test_format_response_unexpected_error(self):
        mock_resp = make_mock_response()
        mock_resp.json.side_effect = RuntimeError('weird error')
        with self.assertRaises(RuntimeError):
            format_response(mock_resp)

    # ----------------------------
    # Dominant emotion tie
    # ----------------------------
    def test_format_response_tie(self):
        mock_json = {'emotionPredictions':[{'emotion': {'anger':0.3,'disgust':0.3,'fear':0.1,'joy':0.2,'sadness':0.1}}]}
        mock_resp = make_mock_response(json_data=mock_json)
        result = format_response(mock_resp)
        self.assertIn(result['dominant_emotion'], ['anger','disgust'])

    # ----------------------------
    # Multiple sentence check
    # ----------------------------
    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_emotion_detector_multiple_sentences(self, mock_post):
        mock_json = {'emotionPredictions':[{'emotion': {'anger':0.1,'disgust':0.1,'fear':0.1,'joy':0.6,'sadness':0.1}}]}
        mock_post.return_value = make_mock_response(json_data=mock_json)
        result = emotion_detector('I am happy. But also worried.')
        self.assertEqual(result['dominant_emotion'], 'joy')


if __name__ == '__main__':
    unittest.main()