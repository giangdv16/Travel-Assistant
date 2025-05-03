#pip install -qU langchain langchain_openai langchain-core langchain-community langgraph psycopg[binary,pool]==3.2.6

import os
with open("clavenueva.txt") as archivo:
  apikey = archivo.read()
os.environ["OPENAI_API_KEY"] = apikey

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage 
from langchain_core.messages import AIMessage 
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import requests
import base64
from io import BytesIO
from IPython.display import display


@tool
def calcularCoordenadas(ciudad: str) -> str:
  """con el nombre de la ciudad retorna una cadena con latitud y longitud separado por comas"""
  url = "https://geocoding-api.open-meteo.com/v1/search?name="+ciudad+"&count=1&language=es"
  response = requests.get(url)
  data =  response.json()
  return str(data['results'][0]['latitude'])+","+ str(data['results'][0]['longitude'])
  if 'results' not in data or not data['results']:
      return "Ciudad no encontrada."

@tool
def calcularHora(latitud: float, longitud: float) -> str:
  """con la latitud y longitud retorna la hora actual"""
  url = "https://api.open-meteo.com/v1/forecast?latitude="+str(latitud)+"&longitude="+str(longitud)+"&current=temperature_2m&timezone=auto"
  response = requests.get(url)
  data =  response.json()
  return data['current']['time']

@tool
def temp(latitud: float, longitud: float) -> float:
    """Con la latitud y longitud retorna la temperatura actual en grados Celsius."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitud}&longitude={longitud}&current=temperature_2m&timezone=auto"
    response = requests.get(url)
    data = response.json()
    temperatura = data['current']['temperature_2m']
    return temperatura

@tool
def destinos(ciudad: str) -> str:
    """con el nombre de la ciudad retorna solo tres sitios turísticos esa ciudad"""
    url = f"https://es.wikipedia.org/w/api.php?action=query&format=json&titles={ciudad}&prop=extracts&exintro=true"
    response = requests.get(url)
    data = response.json()
    
    pages = data['query']['pages']
    page = next(iter(pages.values()))
    
    if 'extract' in page:
        return page['extract']
    else:
        return "No se encontraron detalles sobre esta ciudad."

@tool
def comidas(ciudad: str) -> str:
    """con el nombre de la ciudad retorna solo tres comidas turísticas esa ciudad"""
    url = f"https://es.wikipedia.org/w/api.php?action=query&format=json&titles={ciudad}&prop=extracts&exintro=true"
    response = requests.get(url)
    data = response.json()
    
    pages = data['query']['pages']
    page = next(iter(pages.values()))
    
    if 'extract' in page:
        return page['extract']
    else:
        return "No se encontraron detalles sobre esta ciudad."
    
@tool
def generar_imagen_ciudad(ciudad: str) -> str:
    """Genera una imagen de la ciudad y retorna solo la URL de la imagen generada por DALL·E."""
    import openai
    import os
    openai.api_key = os.environ["OPENAI_API_KEY"]

    prompt = f"Vista de mapa geográfico simple de {ciudad}"

    response = openai.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    return response.data[0].url


model = ChatOpenAI(verbose=True)
memory = MemorySaver()
#construccion del prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """ Eres un asistente de viaje gentil que brinda información de las ciudades, usa solo tus herramientas para responder las preguntas de los usuarios. 
         Solo generarás una imagen con Dall-E si es que el usuario te dice generar imagen.
         Si no cuentas con una herramienta para responder indica que no puedes ayudar con esa consulta .
        """),

     ("human", "{messages}"),

     ]
)
    
toolkit = [calcularCoordenadas,calcularHora,temp,destinos,comidas,generar_imagen_ciudad]
agent = create_react_agent(model, toolkit, checkpointer=memory, prompt=prompt)

historial = []

while True:
    mensaje_usuario = input("Escribe tu consulta (o 'salir' para terminar): ")
    
    # Si el usuario escribe 'salir', termina el bucle
    if mensaje_usuario.lower() == 'salir':
        print("¡Hasta luego!")
        break
    
    
    
    # Ejecutar el agente con la consulta del usuario
    historial.append(HumanMessage(content=mensaje_usuario))
    ultimo = None
    config = {"configurable": {"thread_id": "abc125"}}  # Mantener el contexto si es necesario

    for step in agent.stream(
        {"messages": historial},
        config,
        stream_mode="values",
    ):
        ultimo = step["messages"][-1]

    if isinstance(ultimo, AIMessage):
        print("\n🧠 Respuesta del agente:\n")
        print(ultimo.content)