import streamlit as st
from pdfchatbot import PdfChatbot

# Inicializar la aplicación y cargar configuraciones
chatbot = PdfChatbot("config.yaml")

st.title("RAG PDF Chatbot con Llama 3")
st.write("Carga un archivo PDF y haz preguntas sobre el contenido.")

# Asegurarse de que el historial de la conversación esté inicializado
if 'history' not in st.session_state:
    st.session_state.history = []

# Cargar PDF
pdf_file = st.file_uploader("Sube tu archivo PDF aquí", type=["pdf"])

# Si se carga un PDF, procesarlo
if pdf_file is not None:
    chatbot.load_pdf(pdf_file)
    st.success("PDF cargado y procesado exitosamente.")

    # Preguntas del usuario
    user_question = st.text_input("Haz tu pregunta:")

    if st.button("Enviar"):
        if user_question:
            response = chatbot.answer_question(user_question)
            st.session_state.history.append({"user": user_question, "assistant": response})
            st.write(f"**Assistant:** {response}")
        else:
            st.write("Por favor, ingrese una pregunta.")

    # Mostrar historial de la conversación
    if st.session_state.get('history'):
        st.write("**Historial de conversación:**")
        for entry in st.session_state.history:
            st.write(f"**You:** {entry['user']}")
            st.write(f"**Assistant:** {entry['assistant']}")

    # Botón para copiar la última respuesta
    if st.button("Copiar respuesta"):
        if st.session_state.history:
            st.write(f"**Última respuesta copiada:** {st.session_state.history[-1]['assistant']}")
        else:
            st.write("No hay respuestas para copiar.")

else:
    st.write("Por favor, sube un archivo PDF para comenzar.")
