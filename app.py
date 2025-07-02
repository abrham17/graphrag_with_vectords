import os
import asyncio
import neo4j
import json
from dotenv import load_dotenv
import nest_asyncio
from neo4j_graphrag.llm import MistralAILLM
from neo4j_graphrag.embeddings import MistralAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain_community.vectorstores import FAISS
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from mistralai import Mistral
from langchain.embeddings.base import Embeddings

# Load environment variables
load_dotenv()
URI = "neo4j+s://474fb344.databases.neo4j.io"
AUTH = ("neo4j", "rXazbEHSDzO5Qq3FwQdlKGuN3uy1tYPx9BgvRcbi-o0")
DATABASE = "neo4j"
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY not found in environment variables")

client = Mistral(api_key=api_key)
llm = MistralAILLM(
    model_name="mistral-medium",
    api_key=api_key,
    model_params={"max_tokens": 2000, "response_format": {"type": "json_object"}},
)
embedder = MistralAIEmbeddings()
neo4j_driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)

# Knowledge Graph Construction
async def create_knowledge_graph(file_path, schema=None):
    try:
        kg_builder = SimpleKGPipeline(
            llm=llm,
            driver=neo4j_driver,
            embedder=embedder,
            neo4j_database=DATABASE
        )
        await kg_builder.run_async(file_path=str(file_path))
    except Exception as e:
        st.error(f"Error creating knowledge graph: {str(e)}")
        raise

# Vector Store Creation
def create_vector_store(chunks, embeddings):
    try:
        class WrapperEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [embedder.embed_query(text) for text in texts]
            def embed_query(self, text):
                return embedder.embed_query(text)

        text_embedding_pairs = list(zip(chunks, embeddings))
        vector_store = FAISS.from_embeddings(
            text_embedding_pairs,
            WrapperEmbeddings()
        )
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        raise

# PDF Text Extraction
def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        raise

# Hybrid Retrieval Engine
def vector_retrieval(query, vector_store, top_k=5):
    try:
        results = vector_store.similarity_search_with_score(query, k=top_k)
        output = []
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                doc, score = result
                output.append((doc.page_content, score))
        return output
    except Exception as e:
        st.error(f"Vector retrieval error: {str(e)}")
        return []

def graph_retrieval(query, top_k=5):
    try:
        query_embedding = embedder.embed_query(query)
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        with neo4j_driver.session(database=DATABASE) as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes('node_embedding_index', $top_k, $query_embedding)
                YIELD node, score
                RETURN node, score
                """,
                query_embedding=query_embedding,
                top_k=top_k
            )
            return [(record["node"], record["score"]) for record in result]
    except Exception as e:
        st.error(f"Graph retrieval error: {str(e)}")
        return []

def combine_results(vector_results, graph_results):
    all_results = vector_results + graph_results
    # Normalize scores (assuming vector scores are 0-1, graph scores need scaling if different)
    normalized_results = [(content, min(score, 1.0)) for content, score in all_results]
    # Sort by score and deduplicate
    unique_results = []
    seen = set()
    for content, score in sorted(normalized_results, key=lambda x: x[1], reverse=True):
        content_str = str(content)
        if content_str not in seen:
            unique_results.append((content_str, score))
            seen.add(content_str)
    return unique_results[:5]  # Limit to top 5

# Query Rewriting & Fusion
def rewrite_query(query, num_variants=3):
    try:
        prompt = f"Generate {num_variants} precise variants of this query: {query}"
        # Create a fresh coroutine each time
        response = run_async(llm.ainvoke(prompt))
        variants = response.split("\n") if isinstance(response, str) else [query]
        return [query] + variants[:num_variants]
    except Exception as e:
        st.error(f"Query rewriting error: {str(e)}")
        return [query]

# Answer Generation & Consolidation
def generate_answer(query, retrieved_info):
    info_text = "\n".join([f"[{score:.2f}] {content}" for content, score in retrieved_info])
    prompt = f"""
    Query: {query}
    Information:
    {info_text}
    
    Provide a detailed answer based on the information above. Include source citations (e.g., 'From text chunk' or 'From graph') and handle any conflicting information by noting discrepancies and providing the most likely answer based on evidence.
    """
    try:
        # Create a fresh coroutine each time
        response = run_async(llm.ainvoke(prompt))
        return response
    except Exception as e:
        st.error(f"Answer generation error: {str(e)}")
        return "Unable to generate answer due to an error."

# Sidebar for PDF Upload with Schema Definition
def sidebar_pdf_upload():
    st.sidebar.header("Upload & Process PDF")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")
    schema_input = st.sidebar.text_area("Define Schema (JSON, optional)", placeholder='e.g., {"nodes": ["Person", "Org"], "relationships": ["WORKS_FOR"]}')
    if uploaded_file and st.session_state.get("vector_store") is None:
        with st.spinner("Processing PDF..."):
            try:
                file_path = f"temp_{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                text = extract_text_from_pdf(file_path)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_text(text)
                
                embeddings_response = client.embeddings.create(model="mistral-embed", inputs=chunks)
                embeddings = [data.embedding for data in embeddings_response.data]
                
                vector_store = create_vector_store(chunks, embeddings)
                
                schema = None
                if schema_input.strip():
                    schema = json.loads(schema_input)
                
                asyncio.run(create_knowledge_graph(file_path, schema))
                
                if embeddings and len(embeddings) > 0 and embeddings[0] is not None:
                    dimension = len(embeddings[0])
                else:
                    dimension = 1024
                with neo4j_driver.session(database=DATABASE) as session:
                    result = session.run(
                        "SHOW INDEXES YIELD name WHERE name = 'node_embedding_index' RETURN count(*) AS count"
                    )
                    single_result = result.single()
                    count = single_result['count'] if single_result and 'count' in single_result else 0
                    if count == 0:
                        session.run(
                            """
                            CREATE VECTOR INDEX node_embedding_index IF NOT EXISTS
                            FOR (n:Chunk) ON (n.embedding)
                            OPTIONS {indexConfig: {`vector.dimensions`: $dimension, `vector.similarity_function`: 'cosine'}}
                            """,
                            dimension=dimension
                        )
                
                st.session_state.vector_store = vector_store
                st.session_state.chunks = chunks
                os.remove(file_path)
                st.sidebar.success("PDF processed. You can now chat!")
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")

# Chat Interface
def chat_interface():
    st.title("GraphRAG Q&A System")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    user_input = st.text_input("Ask a question:", "")
    if st.button("Send") and user_input.strip():
        st.session_state.messages.append(("user", user_input))
        with st.spinner("Generating answer..."):
            query_variants = rewrite_query(user_input)
            retrieved_info = []
            for query in query_variants:
                vector_results = vector_retrieval(query, st.session_state.vector_store) if st.session_state.get("vector_store") else []
                graph_results = graph_retrieval(query)
                combined = combine_results(vector_results, graph_results)
                retrieved_info.extend(combined)
            answer = generate_answer(user_input, retrieved_info)
            st.session_state.messages.append(("bot", answer))
    for sender, message in st.session_state.get("messages", []):
        st.markdown(f"**{'You' if sender == 'user' else 'Bot'}:** {message}")

# Main App
def main():
    st.set_page_config(page_title="GraphRAG Q&A", layout="centered")
    sidebar_pdf_upload()
    chat_interface()

def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        # If there's already a running event loop, use it
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in this thread: create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        else:
            return loop.run_until_complete(coro)

if __name__ == "__main__":
    main()