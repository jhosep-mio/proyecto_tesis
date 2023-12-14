# %%
import nltk
import numpy
import tensorflow
import random
import json
from tensorflow.keras.utils import to_categorical

# %%
feedback_entries = []
current_question = None
current_feedback = None

# %%
with open('./data/datasets2.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
# Procesar los datos según tus necesidades
if data:
    # Aquí puedes acceder a las intenciones y respuestas contenidas en data
    intents = data.get('intents', [])
    labels = []
    texts = []

    for intent in intents:
        for pattern in intent.get('patterns', []):
            texts.append(pattern)

        if intent.get('tag') not in labels:
            labels.append(intent.get('tag'))

    # Aquí puedes imprimir o hacer lo que necesites con feedback_entries, labels y texts
    print(feedback_entries)
    print(texts)

# %%
# Generamos el vector de respuestas
# (Cada clase tiene una salida numérica asociada)
output = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        # El identificador de la clase es su índice
        # en la lista de clases o labels
        output.append(labels.index(intent['tag']))

print("Vector de salidas Y:")
print(output)
# Declaramos librería para convertir el vector de salida en una
# matriz categórica

# Generamos la matriz de salidas
train_labels = to_categorical(output, num_classes=len(labels))
print('Matriz de salidas')
print(train_labels)

# %%
import nltk
nltk.download('stopwords')

from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
import re

stop_words = stopwords.words('spanish')

# %%
# Para cada enunciado quitamos las StopWords
# También quitamos los acentos y filtramos signos de puntuación
X = []
for sen in texts:
    sentence = sen
    # Filtrado de stopword
    for stopword in stop_words:
        sentence = sentence.replace(" " + stopword + " ", " ")
    sentence = sentence.replace("á", "a")
    sentence = sentence.replace("é", "e")
    sentence = sentence.replace("í", "i")
    sentence = sentence.replace("ó", "o")
    sentence = sentence.replace("ú", "u")
    # Remover espacios múltiples
    sentence = re.sub(r'\s+', ' ', sentence)
    # Convertir todo a minúsculas
    sentence = sentence.lower()
    # Filtrado de signos de puntuación
    tokenizer = RegexpTokenizer(r'\w+')
    # Tokenización del resultado
    result = tokenizer.tokenize(sentence)
    # Agregar al arreglo los textos "destokenizados" (Como texto nuevamente)
    X.append(TreebankWordDetokenizer().detokenize(result))

# %%
# Importamos la librería para generar la matriz de entrada de textos
# (Importamos pad_sequences y texts_to_sequences para proceso de padding)
from keras.preprocessing.sequence import pad_sequences
# Cantidad de palabras máximas por vector de entrada
maxlen_user = 5
# Preparamos el "molde" para la crear los vectores de secuencia de palabras
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
# Transforma cada texto en una secuencia de valores enteros
X_seq = tokenizer.texts_to_sequences(X)
# Especificamos la matriz (Con padding de posiciones iguales a maxlen)
X_train = pad_sequences(X_seq, padding='post', maxlen=maxlen_user)
print("Matriz de entrada:")
print(X_train)

with open('word_index.json', 'w', encoding='utf-8') as f:
    json.dump(tokenizer.word_index, f, ensure_ascii=False, indent=4)

# %%
from tensorflow.keras.models import load_model

# Cargar el modelo desde el archivo
model = load_model('data/entrenamiento.keras')
predictions = model.predict(X_train)

score = model.evaluate(X_train, train_labels, verbose=1)

# de interrogación y acentos)
def Instancer(inp):
    inp = inp.lower()
    inp = inp.replace("á", "a")
    inp = inp.replace("é", "e")
    inp = inp.replace("í", "i")
    inp = inp.replace("ó", "o")
    inp = inp.replace("ú", "u")
    inp = inp.replace("¿", "")
    inp = inp.replace("?", "")
    txt = [inp]
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=maxlen_user)
    return padded

# %%
# Módulo de detección de gramáticas débiles
Saludos_In = ["Hola", "Holi", "Cómo estás", "Que tal", "Cómo te va"]
Despedidas_In = ["Adios", "Bye", "Hasta luego", "Nos vemos", "Hasta pronto"]
Gracias_In = ["gracias", "te agradezco", "te doy las gracias"]
InsD = [Saludos_In, Despedidas_In, Gracias_In]

Saludos_Out = ["Hola ¿Cómo estás?, te damos la bienvenida a Bacon Grill", "Es un gusto saludarte de nuevo, ¿como puedo ayudarte?", "Hola bienvenido a Bacon Grill, con mucho gusto resolvere todas sus consultas y dudas",
               "Hola bienvenido a Bacon Grill, estoy listo para ayudarte en todo lo que necesites", "Hola bienvenido nuevamente a Bacon Grill, estoy listo para ayudarte a realizar tu compra"]
Despedidas_Out = ["Nos vemos, fue un gusto", "Que te vaya muy bien", "Regresa pronto, adios"]
Gracias_Out = ["Por nada, es un placer", "Me da mucho gusto poder ayudar", "De nada, para eso estoy"]
OutsD = [Saludos_Out, Despedidas_Out, Gracias_Out]

def Weak_grammars(inp):
    index = 0
    weak_act = 0
    response = ''
    for categoria in InsD:
        for gramatica in categoria:
            if inp.lower().count(gramatica.lower()) > 0:
                weak_act = 1
                response = random.choice(OutsD[index])
        index += 1
    return weak_act, response

# %%
# Módulo de detección de gramáticas fuertes
Insultos_In = ["Perra", "Puta", "Estúpida", "Maldita lisiada"]
Fan_In = ["Harry Potter", "Juego de tronos", "El señor de los anillos"]
InsF = [Insultos_In, Fan_In]

Insultos_Out = ["Tú lo serás", "¿Con esa voquita comes?", "Me ofendes, virtualmente hablando"]
Fan_Out = ["Espera, alto, detén todo, platiquemos de esos libros, yo los amo :D",
           "La verdad no escuché lo que dijiste porque yo también soy fan de esos libros :D"]
OutsF = [Insultos_Out, Fan_Out]

def Strong_grammars(inp):
    index = 0
    strong_act = 0
    for categoria in InsF:
        for gramatica in categoria:
            if inp.lower().count(gramatica.lower()) > 0:
                strong_act = 1
                return random.choice(OutsD[index])
        index += 1
    return strong_act

# %%
# Módulo de reconocimiento de entidad País
Paises = {'Parrilla Fire': 'Parrilla Fire', 'Parrilla Af500': 'Parrilla Af500', 'Home Pro': 'Home Pro', 'China re200': 'China re200', 'China 550': 'China 550', 'China Steel': 'China Steel'}
Resp_Paises = ['no cuenta con descuentos disponibles, pero ofrecemos precios competitivos en el mercado.']
Paises_Unknown = ['Actualmente Bacon grill no cuenta con descuentos disponibles, pero ofrecemos precios competitivos en el mercado.',
                'Lo lamento actualmente Bacon grill no cuenta con descuentos disponibles.']

def Country(inp):
    pais_act = 0
    for pais in Paises.keys():
        if inp.lower().count(pais.lower()) > 0:
            pais_act = 1
            return ('\nChatBot: ' + 'Actualmente para el producto '+ str(Paises.get(pais)) + ' Bacon grill ' + random.choice(Resp_Paises) + '[Entidad]\n')
    if pais_act == 0:
        return ('\n' + random.choice(Paises_Unknown) + '\n')

# %%
# Módulo de reconocimiento de entidad Número
import re
import math

Resp_Raiz = ['¿Verdad que soy muy listo?',
             'Soy muy bueno en matemáticas :D',
             'Lo calculé porque un ingeniero me enseñó cómo :D',
             'a los ChatBots nos encantan las matemáticas']
Raiz_Unknown = ['Lo siento pero creo que no me diste un número válido, o soy demasiado torpe para entenderte jeje, ¿Porqué no lo intentas de nuevo?',
                'Mmmm.... No sé si me dijiste el número, pero... no lo entendí, ¿Me lo puedes repetir? y yo lo calculo :)',
                'Creo que no comprendo de la forma que me dijiste el número... ¿O no me lo dijiste?. Inténtalo otra vez']

def Raiz(inp):
    num_act = 0
    num = re.search(r'(\d+)', inp.lower())
    if num != None:
        num_act = 1
        print('\nChatBot: ' + 'La raíz cuadrada de ' + num.group() + ' es ' + str(round(math.sqrt(float(num.group())), 2)) + ' ' + random.choice(Resp_Raiz) + '\n')
    if num_act == 0:
        print('\nChatBot: ' + random.choice(Raiz_Unknown) + '\n')

def identificar_producto(input_usuario):
    productos = ["Fire", "Af500", "Home Pro", "re200", "China 550", "China Steel"]
    input_usuario = input_usuario.lower()
    for producto in productos:
        # Escapamos el nombre del producto por si tiene caracteres que puedan tener significado en una expresión regular
        pattern = re.escape(producto.lower())
        # Buscamos la palabra como una entidad separada, no como subcadena de otra palabra.
        if re.search(r'\b' + pattern + r'\b', input_usuario):
            return producto
    return None

# %%
def clasificar_intent(mensaje):
    # Palabras clave para identificar una intención de pregunta sobre precio
    palabras_clave_precio = ['precio', 'costo', 'cuánto cuesta', 'tarifa', 'valor']
    # Comprobar si alguna de las palabras clave de precio está en el mensaje
    for palabra in palabras_clave_precio:
        if palabra in mensaje:
            return 'precio'
    # Si no se encontró ninguna palabra clave de precio, asumir una intención general
    return 'general'

# %%
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Asumiendo que las funciones como Strong_grammars, Instancer, etc., ya están definidas.

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    request_data = request.get_json()  # Cambiado el nombre de la variable a request_data
    inp = request_data.get("mensaje")
    
    if inp is None:
        return jsonify({"respuesta": "Mensaje no recibido."})
    
    inp = inp.lower()  

    if inp == "salir":
        return jsonify({"respuesta": "Adiós!"})
    
    bot_response = ''  # Inicialización de bot_response
    producto_identificado = identificar_producto(inp)

    if producto_identificado:
            clasificacion = clasificar_intent(inp)
            if clasificacion == 'precio':
            # Código para manejar la respuesta sobre el precio
                precio_tag = f"Precio_{producto_identificado.replace(' ', '_')}"
                for intent in data["intents"]:
                    if intent["tag"].lower() == precio_tag.lower():
                        bot_response = random.choice(intent["responses"])
                        return jsonify({"respuesta": bot_response})
            # Si se identifica un producto, busca el tag y las respuestas asociadas
            for intent in data["intents"]:
                print(intent)
                if producto_identificado.replace(" ", "_").lower() in intent["tag"].lower():
                    # Si el tag coincide, selecciona una respuesta aleatoria y responde
                    bot_response = random.choice(intent["responses"])
                    return jsonify({"respuesta": bot_response})
    

    Strong = Strong_grammars(inp)
    if Strong == 0:
        results = model.predict(Instancer(inp))
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        maxscore = numpy.max(results)
        print(f"Tag predicho: {tag}, Puntaje: {maxscore}")


        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        weak, response = Weak_grammars(inp)
        if weak:
            return jsonify({"respuesta": response})
        if maxscore > 0.5:
            if tag == "Descuentos":
                bot_response = Country(inp)
            else:
                bot_response = str(random.choice(responses))
                print('\nChatBot: ' + str(random.choice(responses)) + ' [' + str(tag) + ']\n')

        else:
            if weak == 0:
                bot_response = 'Lo siento, pero no comprendí, ¿Me puedes preguntar de otra forma?'

    return jsonify({"respuesta": bot_response})

if __name__ == '__main__':
    app.run(debug=False)

