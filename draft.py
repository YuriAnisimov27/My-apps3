import random
import nltk

BOT_CONFIG = {
    'intents': {
        'hello': {
            'examples': ['Привет', 'Здравствуйте', 'Добрый день'],
            'responses': ['Привет, пользователь', 'Юзер, здравствуйте']
        },
        'bye': {
            'examples': ['Пока', 'Доброй ночи'],
            'responses': ['Пока!']
        }
    },
    'failure_phrases': [
        'Я не понял',
        'Переформулируйте, пожалуйста'
    ]
}


def get_intent(text):
    text = text.lower()
    for intent, value in BOT_CONFIG['intents'].items():
        # print(intent, value['examples'])
        for example in value['examples']:
            # print(intent, example)
            distance = nltk.edit_distance(text, example.lower())
            difference = distance / len(example)
            if difference < 0.4:
                return intent


def response_by_intent(intent):
    responses = BOT_CONFIG['intents'][intent]['responses']
    return print(random.choice(responses))


def get_failure_phrase():
    failure_phrases = BOT_CONFIG['failure_phrases']
    return print(random.choice(failure_phrases))


def bot(text):
    intent = get_intent(text)

    if intent:
        return response_by_intent(intent)
    return get_failure_phrase()


while True:
    text = input()
    print(bot(text))
