from flask import Flask, request, Response
from twilio.twiml.messaging_response import MessagingResponse
import os
from dotenv import load_dotenv
import threading 
from twilio.rest import Client 

from mi_rag import initialize, process_documents, setup_vectorstore, setup_rag_system

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER") 

try:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    print("Cliente de Twilio REST inicializado.")
except Exception as e:
    print(f"❌ Error al inicializar el cliente de Twilio: {e}")
    twilio_client = None 

# Inicio el sistema RAG
print("Inicializando sistema RAG para la aplicación web...")
initialize()
documents = process_documents()
db = setup_vectorstore(documents)
rag_chain = setup_rag_system(db)
print("Sistema RAG inicializado y listo para recibir mensajes.")

app = Flask(__name__)

def process_and_reply_async(incoming_msg, from_number, to_number):
    """
    Procesa la pregunta con el RAG y envía la respuesta de vuelta al usuario
    usando la API de Twilio REST. Se ejecuta en un hilo separado.
    """
    if not twilio_client:
        print("❌ Cliente de Twilio no inicializado. No se puede enviar la respuesta asíncrona.")
        return

    try:
        response_text = rag_chain.invoke(incoming_msg)
        
        if not isinstance(response_text, str) or not response_text.strip():
            response_text = "Lo siento, no pude generar una respuesta significativa en este momento."
            print(f"⚠️ La respuesta generada por RAG no es válida o está vacía. Usando mensaje por defecto.")
        
        print(f"🟢 Enviando respuesta asíncrona a {from_number}: {response_text}")

        message = twilio_client.messages.create(
            from_=to_number, 
            to=from_number,  
            body=response_text
        )
        print(f"✅ Mensaje enviado con SID: {message.sid}")

    except Exception as e:
        print(f"❌ Error en el hilo asíncrono al procesar o enviar mensaje: {str(e)}")
        if twilio_client:
            try:
                twilio_client.messages.create(
                    from_=to_number,
                    to=from_number,
                    body="Lo siento, hubo un error al procesar tu solicitud. Por favor, inténtalo de nuevo más tarde."
                )
            except Exception as send_error:
                print(f"❌ Error al enviar mensaje de error al usuario: {send_error}")


@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    """
    Maneja los mensajes de WhatsApp entrantes desde Twilio.
    Esta función responde inmediatamente a Twilio y luego procesa la solicitud
    de forma asíncrona.
    """
    try:
        incoming_msg = request.form.get("Body")
        from_number = request.form.get("From") 
        to_number = request.form.get("To")     

        print(f"📥 Pregunta recibida (desde WhatsApp): {incoming_msg} de {from_number}")

        thread = threading.Thread(
            target=process_and_reply_async,
            args=(incoming_msg, from_number, to_number)
        )
        thread.start()

        twiml_response = MessagingResponse()
        print("✅ Respondiendo a Twilio inmediatamente con 200 OK (respuesta vacía).")
        return Response(str(twiml_response), mimetype='text/xml')

    except Exception as e:
        print(f"❌ Error al procesar el mensaje de WhatsApp (en la función principal): {str(e)}")
        error_response_obj = MessagingResponse()
        error_response_obj.message("Lo siento, hubo un error inicial al procesar tu solicitud.")
        return Response(str(error_response_obj), mimetype='text/xml')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Iniciando la aplicación Flask en http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
