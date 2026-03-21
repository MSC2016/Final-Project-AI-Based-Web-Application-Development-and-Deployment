''' This module uses Watson API's Emotion Predict to analyze the emotions 
    present in a given text.
'''

import requests

# Constants
URL = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
HEADERS = {'grpc-metadata-mm-model-id': 'emotion_aggregated-workflow_lang_en_stock'}
TIMEOUT = 5

def emotion_detector(text_to_analyze : str) -> dict:
    '''
    Predicts emotions from a given text.
    
    Parameters:
                text_to_analyze : str
    Returns:
                dict{
                    'anger': anger_score,
                    'disgust': disgust_score,
                    'fear': fear_score,
                    'joy': joy_score,
                    'sadness': sadness_score,
                    'dominant_emotion': '<name of the dominant emotion>'
                    }
    Raises:
        TypeError: If text_to_analyze is not a string
        ValueError: If text is empty
        TimeoutError: If the API request times out.
        RuntimeError: If the API request faisl for another reason
    '''
    if not isinstance(text_to_analyze, str):
        raise TypeError('input must be a string.')
    if not text_to_analyze.strip():
        raise ValueError('input cannot be empty.')

    payload = {'raw_document' : {'text' : text_to_analyze}}

    try:
        # request emotion prediction to the api
        response = requests.post(URL, json=payload, headers=HEADERS, timeout=TIMEOUT)

        # Handle HTTP code 400
        if response.status_code == 400:
            return {
                'anger': None,
                'disgust': None,
                'fear': None,
                'joy': None,
                'sadness': None,
                'dominant_emotion': None
            }

        # raise exception if status is not in the 200... range
        response.raise_for_status()

        # return the formatted response
        return format_response(response)


    except requests.exceptions.Timeout:
        raise TimeoutError('Request timed out')

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f'API request failed: {e}')

def format_response(response: requests.models.Response) -> dict:
    '''
    Extracts probabilities from the response and adds dominant emotion

    Raises:
        ValueError: if JSON is invalid or structure is unexpected
        RuntimeError: for any other error
    '''
    try:
        # parse JSON response
        data = response.json()
        # extract emotions nested dict
        emotions = data['emotionPredictions'][0]['emotion']
        # finds the dominant emotion
        dominant = max(emotions, key=emotions.get)

        # return a new dict, redundant, but raises exceptions
        # if the API changes and ommits any expected key
        return {'anger': emotions['anger'],
                'disgust': emotions['disgust'],
                'fear': emotions['fear'],
                'joy': emotions['joy'],
                'sadness': emotions['sadness'],
                'dominant_emotion': dominant}

    except ValueError:
        raise ValueError('Invalid JSON response from the API')

    except IndexError:
        raise ValueError('Unexpected response format - Index Error')

    except KeyError:
        raise ValueError('Unexpected response format - Key Error')

    except Exception as e:
        raise RuntimeError(f'Unexpected Error: {e}')
