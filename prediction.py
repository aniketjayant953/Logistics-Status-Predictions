import numpy as np
import re
import pickle
import tensorflow as tf

from keras.utils import pad_sequences

classes = ['Arrival',
           'Departure',
           'Empty Container Released',
           'Empty Return',
           'Gate In',
           'Gate Out',
           'In-transit',
           'Inbound Terminal',
           'Loaded on Vessel',
           'Off Rail',
           'On Rail',
           'Outbound Terminal',
           'Port In',
           'Port Out',
           'Unloaded on Vessel']

with open('tokenizer.pkl', 'rb') as t:
    tokenizer = pickle.load(t)

model = tf.keras.models.load_model('sequential_model.h5')


def clean(text):
    text = text.replace('POL', "Port of Landing")
    text = text.replace('POD', "Port of Discharge")
    text = text.replace('DEPARTCU', "Departure")
    text = text.replace('CY', "Container Yard")
    text = text.replace('CFS', "Container Freight Station")
    text = text.replace('T/S', "TransShipment")

    text = text.replace('/', '')

    not_to_remove = ['TERMINAL', 'OUT', 'PORT', 'IN', 'TOLL', 'PLAZA', 'CROSSED', 'GATE']

    for i in text.split():
        if i in not_to_remove:
            text = text.replace(i, i.lower())

    quote_pattern = r"'([^']*)'"
    text = re.sub(quote_pattern, '', text)

    brackets_pattern = r'\(([^)]*)\)'
    text = re.sub(brackets_pattern, '', text)

    text = re.sub(r'\b[A-Z]+\b', '', text)

    text = re.sub(r'\b\w+\d+\w*\b', '', text)

    return text.strip().lower()


#
#
def predict(text):
    # cleaning the text
    cleaned_text = clean(text)

    # tokenize the data
    text_token = tokenizer.texts_to_sequences([cleaned_text])

    # pad it to the desired dimension
    final_text = pad_sequences(text_token, padding='post', maxlen=57)

    # Predict on the input data
    predictions = model.predict(final_text,verbose=False)

    # Finding the maximum prediction amongst the classes
    predicted_class_index = np.argmax(predictions)

    # all the class list
    class_labels = classes

    # extracting the label from the list
    predicted_class_label = class_labels[predicted_class_index]

    return {text: predicted_class_label}


# print(predict("Unloaded PRSJU"))
