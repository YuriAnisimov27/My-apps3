import random
import nltk
from BIG_CONF import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

BOT_CONFIG_TEST = {
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


# while True:
#     text = input()
#     print(bot(text))

X_text, y = [], []
for intent, value in BOT_CONFIG['intents'].items():
    X_text += value['examples']
    y += [intent] * len(value['examples'])
# print(len(X_text))
# print(len(y))

# Векторизация
vectorizer = CountVectorizer()
# vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)
# print(*vectorizer.get_feature_names())
# print(X.toarray()[0])
# example_vector = vectorizer.transform(['Как дела?']).toarray()[0]

# Класссификация
# clf = LogisticRegression()
# clf.fit(X, y)
# clf.predict([example_vector])
# print(clf.predict([example_vector])[0])
# clf.predict_proba([example_vector])
# print(clf.predict_proba([example_vector])[0])

# Проверка качества
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

scores1 = []
for i in range(10):
    # clf = LogisticRegression()
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    scores1.append(clf.score(X_test, y_test))
print(sum(scores1) / len(scores1))
scores2 = []
for i in range(10):
    clf = LogisticRegression()
    # clf = LinearSVC()
    clf.fit(X_train, y_train)
    scores2.append(clf.score(X_test, y_test))
print(sum(scores2) / len(scores2))

# X_text, y = [], []
# for intent, value in BOT_CONFIG['intents'].items():
#     X_text += value['examples']
#     y += [intent] * len(value['examples'])
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

scores3 = []
for i in range(10):
    # clf = LogisticRegression()
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    scores3.append(clf.score(X_test, y_test))
print(sum(scores3) / len(scores3))
scores4 = []
for i in range(10):
    clf = LogisticRegression()
    # clf = LinearSVC()
    clf.fit(X_train, y_train)
    scores4.append(clf.score(X_test, y_test))
print(sum(scores4) / len(scores4))
