import yaml
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2

class PdfChatbot:
    def __init__(self, config_path):
        # Cargar configuración desde el archivo YAML
        self.config = self.load_config(config_path)
        
        # Crear instancia del cliente OpenAI con la clave y URL base del archivo de configuración
        self.client = OpenAI(api_key=self.config["api_key"], base_url=self.config["base_url"])
        
        # Cargar el modelo de embeddings especificado en la configuración
        self.embeddings_model = SentenceTransformer(self.config["modelEmbeddings"])
        
        # Inicializar variables para almacenar texto del PDF y el índice FAISS
        self.pdf_text = ""
        self.index = None
        self.fragments = []

    def load_config(self, config_path):
        # Cargar y devolver el contenido del archivo de configuración YAML
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def load_pdf(self, pdf_file):
        # Leer el archivo PDF y extraer el texto de cada página
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = []
        for page in range(len(pdf_reader.pages)):
            text.append(pdf_reader.pages[page].extract_text())
        
        # Unir todo el texto extraído y almacenarlo
        self.pdf_text = "\n".join(text)
        
        # Dividir el texto en fragmentos más pequeños y construir el índice FAISS
        self.fragments = self.split_text_into_fragments(self.pdf_text)
        self.index = self.build_faiss_index(self.fragments)

    def split_text_into_fragments(self, text, fragment_size=100):
        # Dividir el texto en fragmentos de tamaño específico
        words = text.split()
        return [" ".join(words[i:i+fragment_size]) for i in range(0, len(words), fragment_size)]

    def build_faiss_index(self, fragments):
        # Crear un índice FAISS utilizando los embeddings de los fragmentos
        embeddings = self.embeddings_model.encode(fragments)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings, dtype=np.float32))
        return index

    def find_relevant_fragment(self, query, top_k=5):
        # Encontrar los fragmentos más relevantes para la consulta usando FAISS
        query_embedding = self.embeddings_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)
        return [self.fragments[idx] for idx in indices[0]]

    def query_model(self, context, query):
        # Enviar una consulta al modelo de lenguaje usando el cliente OpenAI
        messages = [
            {"role": "system", "content": "You are an AI assistant who knows everything."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
        response = self.client.chat.completions.create(
            model=self.config["autoModelForCausalLM"],
            messages=messages
        )
        
        # Acceder correctamente a la respuesta generada por el modelo
        answer = response.choices[0].message.content
        return answer

    def answer_question(self, query):
        # Encontrar fragmentos relevantes en el PDF y usar el modelo para responder
        relevant_fragments = self.find_relevant_fragment(query)
        context = " ".join(relevant_fragments)
        answer = self.query_model(context, query)
        
        # Añadir una nota si no se encontró información relevante
        if not context.strip():
            answer += "\nNota: No se encontró información relevante en el PDF. Considera consultar otras fuentes para investigar más sobre este tema."
        return answer

# Configuración de la ruta al archivo config.yaml
config_path = "config.yaml"

# Crear instancia de PdfChatbot con la ruta de configuración
chatbot = PdfChatbot(config_path)
