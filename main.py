import empath
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
import pandas as pd

## ANÁLISIS DE SENTIMIENTOS Y DETECCIÓN DE EMOCIONES
def vader_dic(frame_el_tiempo, frame_el_espectador, frame_semana):
    dic_eltiempo = {'Negativos' : 0, 'Neutros': 0, 'Positivos': 0}
    dic_semana = {'Negativos': 0, 'Neutros': 0, 'Positivos': 0}
    dic_elespectador = {'Negativos': 0, 'Neutros': 0, 'Positivos': 0}
    # Recorrer por filas el tiempo

    for indice, fila in frame_el_tiempo.iterrows():
        contenido = frame_el_tiempo.iat[indice, 2]
        emotion = vader(contenido)
        if emotion == 1:
            dic_eltiempo['Positivos'] = dic_eltiempo['Positivos'] + 1
        elif emotion == -1:
            dic_eltiempo['Negativos'] = dic_eltiempo['Negativos'] + 1
        else:
            dic_eltiempo['Neutros'] = dic_eltiempo['Neutros'] + 1
    # Recorrer por filas semana
    for indice, fila in frame_semana.iterrows():
        contenido = frame_semana.iat[indice, 2]
        emotion = vader(contenido)
        if emotion == 1:
            dic_semana['Positivos'] = dic_semana['Positivos'] + 1
        elif emotion == -1:
            dic_semana['Negativos'] = dic_semana['Negativos'] + 1
        else:
            dic_semana['Neutros'] = dic_semana['Neutros'] + 1
    # Recorrer por filas el tiempo
    for indice, fila in frame_el_espectador.iterrows():
        contenido = frame_el_espectador.iat[indice, 2]
        emotion = vader(contenido)
        if emotion == 1:
            dic_elespectador['Positivos'] = dic_elespectador['Positivos'] + 1
        elif emotion == -1:
            dic_elespectador['Negativos'] = dic_elespectador['Negativos'] + 1
        else:
            dic_elespectador['Neutros'] = dic_elespectador['Neutros'] + 1
    dic_periodicos = {'elTiempo': dic_eltiempo, 'semana':dic_semana, 'elEspectador':dic_elespectador}
    return dic_periodicos

def vader(texto):
    sid = SentimentIntensityAnalyzer()
    sentimiento = sid.polarity_scores(texto)
    polarity_score = sentimiento['compound']
    if polarity_score >= 0.33:
        emotion = 1
    elif polarity_score <= -0.33:
        emotion = -1
    else:
        emotion = 0
    return emotion

def textblob_dic(frame_el_tiempo, frame_el_espectador, frame_semana):
    dic_eltiempo = {'Negativos' : 0, 'Neutros': 0, 'Positivos': 0}
    dic_semana = {'Negativos': 0, 'Neutros': 0, 'Positivos': 0}
    dic_elespectador = {'Negativos': 0, 'Neutros': 0, 'Positivos': 0}
    # Recorrer por filas el tiempo

    for indice, fila in frame_el_tiempo.iterrows():
        contenido = frame_el_tiempo.iat[indice, 2]
        emotion = textblob_f(contenido)
        if emotion == 1:
            dic_eltiempo['Positivos'] = dic_eltiempo['Positivos'] + 1
        elif emotion == -1:
            dic_eltiempo['Negativos'] = dic_eltiempo['Negativos'] + 1
        else:
            dic_eltiempo['Neutros'] = dic_eltiempo['Neutros'] + 1
    # Recorrer por filas semana
    for indice, fila in frame_semana.iterrows():
        contenido = frame_semana.iat[indice, 2]
        emotion = textblob_f(contenido)
        if emotion == 1:
            dic_semana['Positivos'] = dic_semana['Positivos'] + 1
        elif emotion == -1:
            dic_semana['Negativos'] = dic_semana['Negativos'] + 1
        else:
            dic_semana['Neutros'] = dic_semana['Neutros'] + 1
    # Recorrer por filas el tiempo
    for indice, fila in frame_el_espectador.iterrows():
        contenido = frame_el_espectador.iat[indice, 2]
        emotion = textblob_f(contenido)
        if emotion == 1:
            dic_elespectador['Positivos'] = dic_elespectador['Positivos'] + 1
        elif emotion == -1:
            dic_elespectador['Negativos'] = dic_elespectador['Negativos'] + 1
        else:
            dic_elespectador['Neutros'] = dic_elespectador['Neutros'] + 1
    dic_periodicos = {'elTiempo': dic_eltiempo, 'semana':dic_semana, 'elEspectador':dic_elespectador}
    return dic_periodicos

def textblob_f(texto):
    blob = TextBlob(texto)

    # Obtener la puntuación de sentimiento
    sentiment_score = blob.sentiment.polarity

    # obtener la puntuacón de la subjetividad
    subjectivity = blob.sentiment.subjectivity

    # Determinar la emoción basada en la puntuación de sentimiento
    if sentiment_score >= 0.33:
        emotion = 1
    elif sentiment_score <= -0.33:
        emotion = -1
    else:
        emotion = 0

    return emotion

def empath_dic(frame_el_tiempo, frame_el_espectador, frame_semana):
    dic_eltiempo = {"surprise": 0, "fear": 0, "love": 0, "sadness":0, "anticipation": 0, "optimism": 0}
    dic_semana = {"surprise": 0, "fear": 0, "love": 0, "sadness":0, "anticipation": 0, "optimism": 0}
    dic_elespectador = {"surprise": 0, "fear": 0, "love": 0, "sadness":0, "anticipation": 0, "optimism": 0}
    # Recorrer por filas el tiempo

    for indice, fila in frame_el_tiempo.iterrows():
        contenido = frame_el_tiempo.iat[indice, 2]
        emotion = empath_f(contenido)
        if emotion == "surprise":
            dic_eltiempo['surprise'] = dic_eltiempo['surprise'] + 1
        elif emotion == "fear":
            dic_eltiempo['fear'] = dic_eltiempo['fear'] + 1
        elif emotion == "love":
            dic_eltiempo['love'] = dic_eltiempo['love'] + 1
        elif emotion == "sadness":
            dic_eltiempo['sadness'] = dic_eltiempo['sadness'] + 1
        elif emotion == "anticipation":
            dic_eltiempo['anticipation'] = dic_eltiempo['anticipation'] + 1
        else:
            dic_eltiempo['optimism'] = dic_eltiempo['optimism'] + 1
    # Recorrer por filas semana
    for indice, fila in frame_semana.iterrows():
        contenido = frame_semana.iat[indice, 2]
        emotion = empath_f(contenido)
        if emotion == "surprise":
            dic_semana['surprise'] = dic_semana['surprise'] + 1
        elif emotion == "fear":
            dic_semana['fear'] = dic_semana['fear'] + 1
        elif emotion == "love":
            dic_semana['love'] = dic_semana['love'] + 1
        elif emotion == "sadness":
            dic_semana['sadness'] = dic_semana['sadness'] + 1
        elif emotion == "anticipation":
            dic_semana['anticipation'] = dic_semana['anticipation'] + 1
        else:
            dic_semana['optimism'] = dic_semana['optimism'] + 1
    # Recorrer por filas el tiempo
    for indice, fila in frame_el_espectador.iterrows():
        contenido = frame_el_espectador.iat[indice, 2]
        emotion = empath_f(contenido)
        if emotion == "surprise":
            dic_elespectador['surprise'] = dic_elespectador['surprise'] + 1
        elif emotion == "fear":
            dic_elespectador['fear'] = dic_elespectador['fear'] + 1
        elif emotion == "love":
            dic_elespectador['love'] = dic_elespectador['love'] + 1
        elif emotion == "sadness":
            dic_elespectador['sadness'] = dic_elespectador['sadness'] + 1
        elif emotion == "anticipation":
            dic_elespectador['anticipation'] = dic_elespectador['anticipation'] + 1
        else:
            dic_elespectador['optimism'] = dic_elespectador['optimism'] + 1
    dic_periodicos = {'elTiempo': dic_eltiempo, 'semana': dic_semana, 'elEspectador': dic_elespectador}
    return dic_periodicos

def empath_f(texto):
    lexicon = empath.Empath()
    emotions = lexicon.analyze(texto, normalize=True)
    surprise_score = emotions["surprise"]
    fear_score = emotions["fear"]
    love_score = emotions["love"]
    optimism_score = emotions["optimism"]
    anticipation_score = emotions["anticipation"]
    sadness_score = emotions["sadness"]
    dic_emotions = {"surprise": surprise_score, "fear": fear_score, "love": love_score, "sadness":sadness_score, "anticipation": anticipation_score, "optimism": optimism_score}
    # Obtener la clave del elemento más alto
    emotion = max(dic_emotions, key=dic_emotions.get)

    # Imprimir la clave del elemento más alto
    return emotion

def nrclex_dic(frame_el_tiempo, frame_el_espectador, frame_semana):
    dic_eltiempo = {"fear": 0, "negative": 0, "surprise": 0, "positive": 0, "trust": 0, "disgust": 0, "anticipation": 0, "anger": 0}
    dic_semana = {"fear": 0, "negative": 0, "surprise": 0, "positive": 0, "trust": 0, "disgust": 0, "anticipation": 0, "anger": 0}
    dic_elespectador = {"fear": 0, "negative": 0, "surprise": 0, "positive": 0, "trust": 0, "disgust": 0, "anticipation": 0, "anger": 0}
    # Recorrer por filas el tiempo

    for indice, fila in frame_el_tiempo.iterrows():
        contenido = frame_el_tiempo.iat[indice, 2]
        emotion = nrclex_f(contenido)
        if emotion == "fear":
            dic_eltiempo['fear'] = dic_eltiempo['fear'] + 1
        elif emotion == "negative":
            dic_eltiempo['negative'] = dic_eltiempo['negative'] + 1
        elif emotion == "surprise":
            dic_eltiempo['surprise'] = dic_eltiempo['surprise'] + 1
        elif emotion == "positive":
            dic_eltiempo['positive'] = dic_eltiempo['positive'] + 1
        elif emotion == "trust":
            dic_eltiempo['trust'] = dic_eltiempo['trust'] + 1
        elif emotion == "disgust":
            dic_eltiempo['disgust'] = dic_eltiempo['disgust'] + 1
        elif emotion == "anticipation":
            dic_eltiempo['anticipation'] = dic_eltiempo['anticipation'] + 1
        else:
            dic_eltiempo['anger'] = dic_eltiempo['anger'] + 1
    # Recorrer por filas semana
    for indice, fila in frame_semana.iterrows():
        contenido = frame_semana.iat[indice, 2]
        emotion = nrclex_f(contenido)
        if emotion == "fear":
            dic_semana['fear'] = dic_semana['fear'] + 1
        elif emotion == "negative":
            dic_semana['negative'] = dic_semana['negative'] + 1
        elif emotion == "surprise":
            dic_semana['surprise'] = dic_semana['surprise'] + 1
        elif emotion == "positive":
            dic_semana['positive'] = dic_semana['positive'] + 1
        elif emotion == "trust":
            dic_semana['trust'] = dic_semana['trust'] + 1
        elif emotion == "disgust":
            dic_semana['disgust'] = dic_semana['disgust'] + 1
        elif emotion == "anticipation":
            dic_semana['anticipation'] = dic_semana['anticipation'] + 1
        else:
            dic_semana['anger'] = dic_semana['anger'] + 1
    # Recorrer por filas el tiempo
    for indice, fila in frame_el_espectador.iterrows():
        contenido = frame_el_espectador.iat[indice, 2]
        emotion = nrclex_f(contenido)
        if emotion == "fear":
            dic_elespectador['fear'] = dic_elespectador['fear'] + 1
        elif emotion == "negative":
            dic_elespectador['negative'] = dic_elespectador['negative'] + 1
        elif emotion == "surprise":
            dic_elespectador['surprise'] = dic_elespectador['surprise'] + 1
        elif emotion == "positive":
            dic_elespectador['positive'] = dic_elespectador['positive'] + 1
        elif emotion == "trust":
            dic_elespectador['trust'] = dic_elespectador['trust'] + 1
        elif emotion == "disgust":
            dic_elespectador['disgust'] = dic_elespectador['disgust'] + 1
        elif emotion == "anticipation":
            dic_elespectador['anticipation'] = dic_elespectador['anticipation'] + 1
        else:
            dic_elespectador['anger'] = dic_elespectador['anger'] + 1

    dic_periodicos = {'elTiempo': dic_eltiempo, 'semana': dic_semana, 'elEspectador': dic_elespectador}
    return dic_periodicos

def nrclex_f(texto):
    text_object = NRCLex(texto)
    emotions = text_object.raw_emotion_scores
    emotion = max(emotions, key=emotions.get)
    return emotion

# Leer el archivo Excel
data_frame_el_espectador = pd.read_excel('el_espectador.xlsx')
data_frame_el_tiempo = pd.read_excel('eltiempo.xlsx')
data_frame_semana = pd.read_excel('semana.xlsx')

#Análisis de sentimientos
print("VADER")
print("-----")
vader_result = vader_dic(data_frame_el_tiempo, data_frame_el_espectador, data_frame_semana)
print("El Tiempo: " + str(vader_result.get("elTiempo")))
print("Semana: " + str(vader_result.get("semana")))
print("El Espectador: " + str(vader_result.get("elEspectador")))
print("\n")

print("TEXT_BLOB")
print("---------")
TextBlob_result = textblob_dic(data_frame_el_tiempo, data_frame_el_espectador, data_frame_semana)
print("El Tiempo: " + str(TextBlob_result.get("elTiempo")))
print("Semana: " + str(TextBlob_result.get("semana")))
print("El Espectador: " + str(TextBlob_result.get("elEspectador")))
print("\n")

#detección emocional
print("NRCLEX")
print("-------")
nrclex_result = nrclex_dic(data_frame_el_tiempo, data_frame_el_espectador, data_frame_semana)
print("El Tiempo: " + str(nrclex_result.get("elTiempo")))
print("Semana: " + str(nrclex_result.get("semana")))
print("El Espectador: " + str(nrclex_result.get("elEspectador")))
print("\n")

print("EMPATH")
print("-----")
empath_result = empath_dic(data_frame_el_tiempo, data_frame_el_espectador, data_frame_semana)
print("El Tiempo: " + str(empath_result.get("elTiempo")))
print("Semana: " + str(empath_result.get("semana")))
print("El Espectador: " + str(empath_result.get("elEspectador")))