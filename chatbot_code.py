import transformers
from transformers import DPRContextEncoderTokenizer, DPRContextEncoder, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch
import numpy as np
from google import genai
import chromadb
from chromadb.utils import embedding_functions
import uuid
from dotenv import load_dotenv
import os


load_dotenv()
gemini_api_key = os.environ["GEMINI_API_KEY"]

class TravelServiceChatbot:
    def __init__(self, gemini_api_key, collection_name="travel_knowledge_base"):
        # Initialize encoders and tokenizers for RAG
        self.context_model_name = 'facebook/dpr-ctx_encoder-single-nq-base'
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(self.context_model_name)
        self.context_encoder = DPRContextEncoder.from_pretrained(self.context_model_name)

        self.question_encoder_model_name = 'facebook/dpr-question_encoder-single-nq-base'
        self.question_encoder = DPRQuestionEncoder.from_pretrained(self.question_encoder_model_name)
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(self.question_encoder_model_name)

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma")

        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Travel company knowledge base"}
        )

        # Initialize Gemini LLM
        self.gemini_api_key = gemini_api_key
        self.llm_client = genai.Client(api_key=self.gemini_api_key)

        self.conversation_history = []  # This will store tuples of (user_message, agent_response)


    def _encode_text(self, text):
        """Encode a single text using DPR context encoder"""
        inputs = self.context_tokenizer(text, max_length=512, truncation=True,
                                      padding='max_length', return_tensors="pt")
        outputs = self.context_encoder(**inputs)
        return outputs.pooler_output.detach().numpy()[0]  # Return as numpy array

    def load_knowledge_base(self, content, chunk_size=512):
        """Load and process company knowledge base from text content into ChromaDB"""
        # Split the content into paragraphs
        paragraphs = content.split('\n\n')
        paragraphs = [para.strip() for para in paragraphs if len(para.strip()) > 0]

        # Prepare data for ChromaDB
        embeddings = []
        documents = []
        ids = []
        metadata = []

        for para in paragraphs:
            # Generate embedding
            embedding = self._encode_text(para)

            # Prepare document data
            doc_id = str(uuid.uuid4())

            embeddings.append(embedding)
            documents.append(para)
            ids.append(doc_id)
            metadata.append({"source": "uploaded_content", "length": len(para)})

        # Add to ChromaDB collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=ids,
            metadatas=metadata
        )

        return len(paragraphs)

    def _encode_question(self, question):
        """Encode a question using DPR question encoder"""
        question_inputs = self.question_tokenizer(question, return_tensors='pt')
        question_embedding = self.question_encoder(**question_inputs).pooler_output.detach().numpy()[0]
        return question_embedding

    def _search_relevant_contexts(self, question, k=3):
        """Search for relevant context given a question using ChromaDB"""
        # Encode the question
        question_embedding = self._encode_question(question)

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=k
        )

        return results['documents'][0]  # Return list of relevant documents

    def generate_response(self, question, relevant_contexts,history_text):
        """Generate response using Gemini LLM"""
        prompt = f"""You are a helpful travel company customer service agent. Use the following context to answer the customer's question.
        If the answer cannot be found in the context, politely say so and provide general travel-related assistance.

        Conversation History:

        {history_text}

        Context:
        {' '.join(relevant_contexts)}

        Customer Question: {question}

        Please provide a helpful, professional response, focusing on the following guidelines:
        1. Be concise but informative
        2. Use a friendly, professional tone
        3. If suggesting something not in the context, clearly indicate it's a general suggestion
        4. Always prioritize accuracy over speculation
        5. Use conversation history to retrieve information if needed.

        Response:"""

        response = self.llm_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )

        return response.text

    def chat(self, question):
        """Main chat interface with conversation history"""
        # Append the new user question to the history
        self.conversation_history.append(("User", question))

        # Retrieve relevant context (from the knowledge base)
        relevant_contexts = self._search_relevant_contexts(question)

        # Build conversation context string (limit to the last 20 messages)
        recent_history = self.conversation_history[-20:]  # Get the last 20 turns
        history_text = ""
        for role, message in recent_history:
            history_text += f"{role}: {message}\n"

        # Generate the response with history included in the prompt
        response = self.generate_response(question, relevant_contexts, history_text)

        # Append the agent response to the conversation history
        self.conversation_history.append(("Agent", response))

        return response

    def add_knowledge(self, text, metadata=None):
        """Add new knowledge to the existing knowledge base"""
        doc_id = str(uuid.uuid4())
        embedding = self._encode_text(text)

        if metadata is None:
            metadata = {"source": "manual_addition", "length": len(text)}

        self.collection.add(
            embeddings=[embedding],
            documents=[text],
            ids=[doc_id],
            metadatas=[metadata]
        )

        return doc_id

def main():
    # Initialize chatbot
    GEMINI_API_KEY = gemini_api_key
    chatbot = TravelServiceChatbot(GEMINI_API_KEY)

    # Load initial knowledge base if needed
    kb_file = "travel_company_kb.txt"
    try:
        num_paragraphs = chatbot.load_knowledge_base(kb_file)
        print(f"Loaded {num_paragraphs} paragraphs into knowledge base")
    except FileNotFoundError:
        print("No knowledge base file found. Starting with empty knowledge base.")

    # Chat loop
    print("Travel Company Customer Service Bot (type 'exit' to quit)")
    print("Admin commands: !add <text> (adds new knowledge)")
    print("-" * 50)

    while True:
        user_input = input("\nCustomer: ").strip()

        if user_input.lower() == 'exit':
            break

        # Check for admin commands
        if user_input.startswith('!add '):
            new_knowledge = user_input[5:].strip()
            doc_id = chatbot.add_knowledge(new_knowledge)
            print(f"\nSystem: Added new knowledge with ID: {doc_id}")
            continue

        response = chatbot.chat(user_input)
        print("\nAgent:", response)

if __name__ == "__main__":
    main()