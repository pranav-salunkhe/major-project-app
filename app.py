import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Neo4jVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


from neo4j import GraphDatabase

# Page configuration
st.set_page_config(
    page_title="MIMIC-III Database Chatbot",
    page_icon="🏥",
    layout="wide"
)


# App title and description
st.title("🏥 MIMIC-III Database Query Assistant")
st.markdown("""
This chatbot allows you to query the MIMIC-III healthcare database using natural language.
Simply type your question about patients, diagnoses, admissions, or other MIMIC-III data.
""")

# Sidebar for API credentials
with st.sidebar:
    st.header("Configuration")
    
        # Get credentials from secrets or show placeholder fields for local development
    if "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        st.success("OpenAI API key loaded from secrets!")
    else:
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        st.warning("API key not found in secrets. Enter it manually or add to .streamlit/secrets.toml")
    
    # Neo4j credentials
    st.subheader("Neo4j Database Credentials")
    
    # Get Neo4j credentials from secrets or show placeholder fields
    if all(k in st.secrets for k in ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"]):
        neo4j_uri = st.secrets["NEO4J_URI"]
        neo4j_user = st.secrets["NEO4J_USER"]
        neo4j_password = st.secrets["NEO4J_PASSWORD"]
        st.success("Neo4j credentials loaded from secrets!")
    else:
        neo4j_uri = st.text_input("Neo4j URI", placeholder="bolt://localhost:7687")
        neo4j_user = st.text_input("Neo4j Username", placeholder="neo4j")
        neo4j_password = st.text_input("Neo4j Password", type="password")
        st.warning("Neo4j credentials not found in secrets. Enter manually or add to .streamlit/secrets.toml")
    
    # # OpenAI API Key
    # openai_api_key = st.text_input("OpenAI API Key", type="password")
    
    # # Neo4j credentials
    # st.subheader("Neo4j Database Credentials")
    # neo4j_uri = st.text_input("Neo4j URI", placeholder="bolt://localhost:7687")
    # neo4j_user = st.text_input("Neo4j Username", placeholder="neo4j")
    # neo4j_password = st.text_input("Neo4j Password", type="password")
    
    # Query approach selection
    query_method = st.radio(
        "Query Method",
        ["Cypher Generation", "Embedding-based QA", "Hybrid Approach"]
    )
    
    # Information about the database
    st.subheader("About MIMIC-III")
    st.info("""
    MIMIC-III (Medical Information Mart for Intensive Care III) is a large, 
    freely-available database comprising deidentified health-related data associated 
    with over 40,000 patients who stayed in critical care units.
    """)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to validate and establish Neo4j connection
def test_neo4j_connection(uri, user, password):
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            # Simple query to test connection
            result = session.run("RETURN 'Connection successful' as message")
            message = result.single()["message"]
        driver.close()
        return True, message
    except Exception as e:
        return False, str(e)

# Function for Cypher-based question answering
def cypher_qa(user_question, graph, llm):
    # Define Cypher generation prompt
    cypher_prompt = PromptTemplate(
        template="""
        You are an expert Neo4j Developer translating user questions into Cypher to answer questions about MIMIC-III database tables.
        Convert the user's question based on the schema.
        ----------------
        Schema: {schema}
        Question: {question}
        """,
        input_variables=["schema", "question"],
    )
    
    # Create the chain
    cypher_chain = GraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        cypher_prompt=cypher_prompt,
        verbose=True,
        allow_dangerous_requests=True,
        return_intermediate_steps=True,
        return_direct=False,
    )
    
    # Get response
    result = cypher_chain.invoke({"query": user_question})
    
    # Enhanced error handling - check if the response is too generic
    if "I'm sorry" in result["result"] and "can't determine" in result["result"]:
        # Extract the generated Cypher and query results
        cypher_query = result["intermediate_steps"]["cypher"]
        context = result["intermediate_steps"]["context"]
        
        if not context or (isinstance(context, list) and len(context) == 0):
            return f"""No data was returned for this query. The generated Cypher was:
            
```
{cypher_query}
```

This could mean either:
1. The query is asking about data that doesn't exist in the database
2. The Cypher query needs refinement to match the database schema
3. The relationship being queried may use different node labels or property names

Try rephrasing your question to be more specific about the entities in the MIMIC-III database."""
        
        # Try to get a better response by passing the raw data to a final prompt
        better_response_prompt = ChatPromptTemplate.from_template(
            """You are an expert in healthcare data analysis working with the MIMIC-III database.
            
            User Question: {question}
            
            Raw Database Results: {context}
            
            Please provide a clear, detailed answer to the user's question based on these database results.
            If the results don't fully answer the question, explain what can be determined and what cannot.
            """
        )
        
        better_response_chain = (
            {"question": RunnablePassthrough(), "context": lambda _: str(context)}
            | better_response_prompt
            | llm
            | StrOutputParser()
        )
        
        return better_response_chain.invoke(user_question)
    
    return result["result"]

# Function for embedding-based question answering
def embedding_qa(user_question, uri, user, password, llm, embeddings):
    try:
        # Initialize Neo4j Vector store with embeddings
        vector_store = Neo4jVector.from_existing_index(
            embeddings=embeddings,
            url=uri,
            username=user,
            password=password,
            index_name="mimic_document_embeddings",  
            node_label="Document",                 
            text_node_property="text",              
            embedding_node_property="embedding"     
        )
        
        # Retrieve relevant documents
        docs = vector_store.similarity_search(user_question, k=3)
        
        if not docs:
            return "No relevant information found in the database. Try a different question about MIMIC-III data."
        
        # Create QA chain
        qa_prompt = ChatPromptTemplate.from_template(
            """You are a healthcare data expert working with the MIMIC-III clinical database.
            
            Answer the question based only on the following context about MIMIC-III data:
            
            {context}
            
            Question: {question}
            
            If the context doesn't contain relevant information to answer the question completely, 
            explain what you can determine from the context and what information is missing.
            """
        )
        
        qa_chain = (
            {"context": lambda input_dict: "\n\n".join([doc.page_content for doc in input_dict["docs"]]), 
             "question": lambda input_dict: input_dict["question"]}
            | qa_prompt
            | llm
            | StrOutputParser()
        )
        
        # Get response
        return qa_chain.invoke({"docs": docs, "question": user_question})
        
    except Exception as e:
        return f"Error with embedding-based search: {str(e)}\n\nPlease check that your Neo4j database has vector embeddings set up properly with an index named 'mimic_document_embeddings'."

# Function to handle hybrid approach
def hybrid_qa(user_question, graph, uri, user, password, llm, embeddings):
    # Try Cypher approach first
    cypher_result = cypher_qa(user_question, graph, llm)
    
    # If Cypher result seems generic or empty, try embedding approach
    if "I'm sorry" in cypher_result and "can't determine" in cypher_result:
        embedding_result = embedding_qa(user_question, uri, user, password, llm, embeddings)
        
        # Use a final prompt to integrate both results
        integration_prompt = ChatPromptTemplate.from_template(
            """You are a healthcare data expert working with the MIMIC-III clinical database.
            
            User Question: {question}
            
            Result from database query: {cypher_result}
            
            Result from knowledge base: {embedding_result}
            
            Please provide the most complete and accurate answer by combining both sources of information.
            Focus on giving the user specific, accurate information from the actual database results.
            """
        )
        
        integration_chain = (
            {"question": RunnablePassthrough(), 
             "cypher_result": lambda _: cypher_result,
             "embedding_result": lambda _: embedding_result}
            | integration_prompt
            | llm
            | StrOutputParser()
        )
        
        return integration_chain.invoke(user_question)
    
    return cypher_result

# Initialize Neo4j schema for display
def get_schema_info(graph):
    try:
        schema = graph.get_schema
        return schema
    except:
        return "Failed to retrieve schema. Please check database connection."

# Function to process the query
def process_query(user_question):
    try:
        # Check Neo4j connection
        connection_successful, message = test_neo4j_connection(neo4j_uri, neo4j_user, neo4j_password)
        if not connection_successful:
            return f"Database connection error: {message}. Please check your Neo4j credentials."
        
        # Initialize LLM
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            temperature=0,
            model="gpt-4"
        )
        
        # Initialize Neo4j graph
        graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password
        )
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key
        )
        
        # Process query based on selected method
        if query_method == "Cypher Generation":
            return cypher_qa(user_question, graph, llm)
        elif query_method == "Embedding-based QA":
            return embedding_qa(user_question, neo4j_uri, neo4j_user, neo4j_password, llm, embeddings)
        else:  # Hybrid approach
            return hybrid_qa(user_question, graph, neo4j_uri, neo4j_user, neo4j_password, llm, embeddings)
    
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Display database schema button
if st.sidebar.button("View Database Schema"):
    if not neo4j_uri or not neo4j_user or not neo4j_password:
        st.sidebar.error("Please provide Neo4j credentials first")
    else:
        try:
            graph = Neo4jGraph(
                url=neo4j_uri,
                username=neo4j_user,
                password=neo4j_password
            )
            schema = get_schema_info(graph)
            st.sidebar.code(schema, language="json")
        except Exception as e:
            st.sidebar.error(f"Error retrieving schema: {str(e)}")

# Helper function to create vector embeddings (for setup instructions)
# def setup_vector_embeddings():
#     setup_instructions = """
#     ## Setting Up Vector Embeddings in Neo4j
    
#     To use embedding-based question answering, you need to create document embeddings in your Neo4j database:
    
#     ```python
#     from langchain_openai import OpenAIEmbeddings
#     from langchain_community.vectorstores import Neo4jVector
    
#     # 1. Prepare your documents
#     # Extract key information from your MIMIC-III database as documents
#     documents = [
#         # Example: Convert patient records to document format
#         # {"text": "Patient 123 was diagnosed with heart failure and stayed for 5 days", "metadata": {...}}
#     ]
    
#     # 2. Initialize embeddings
#     embeddings = OpenAIEmbeddings(openai_api_key="your-key")
    
#     # 3. Create vector store in Neo4j
#     Neo4jVector.from_documents(
#         documents,
#         embeddings,
#         url="your-neo4j-uri",
#         username="neo4j",
#         password="password",
#         index_name="mimic_document_embeddings",
#         node_label="Document",
#     )
#     ```
    
#     For detailed instructions, refer to the LangChain documentation on Neo4jVector.
#     """
#     st.sidebar.expander("How to Set Up Vector Embeddings").markdown(setup_instructions)

# Add the setup instructions to sidebar
# setup_vector_embeddings()

# Chat input
prompt = st.chat_input("Ask a question about the MIMIC-III database...")

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        # Check if credentials are provided
        if not openai_api_key or not neo4j_uri or not neo4j_user or not neo4j_password:
            response = "Please provide all required API keys and database credentials in the sidebar."
            st.markdown(response)
        else:
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            # Process the query
            response = process_query(prompt)
            
            # Display the response
            message_placeholder.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Example queries
st.subheader("Example Queries")
example_queries = [
    "Which patients have more than two diagnoses?",
    "What is the average length of stay for patients with heart failure?",
    "Show me patients who have been readmitted within 30 days",
    "Which diagnosis is most common among elderly patients?",
    "How many patients have both diabetes and hypertension?"
]

for example in example_queries:
    if st.button(example):
        # Add example query to chat history
        st.session_state.messages.append({"role": "user", "content": example})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(example)
        
        # Display assistant response
        with st.chat_message("assistant"):
            if not openai_api_key or not neo4j_uri or not neo4j_user or not neo4j_password:
                response = "Please provide all required API keys and database credentials in the sidebar."
                st.markdown(response)
            else:
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                
                # Process the query
                response = process_query(example)
                
                # Display the response
                message_placeholder.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer with setup instructions for vector embeddings
# st.markdown("---")
# with st.expander("Setting Up Vector Embeddings for Better Results"):
#     st.markdown("""
#     To get the most out of this chatbot, you should set up vector embeddings in your Neo4j database.
#     This allows the system to find relevant information even when questions don't directly match the database schema.
    
#     ## Basic Steps:
    
#     1. Extract meaningful text from your MIMIC-III database (patient summaries, diagnosis descriptions, etc.)
#     2. Create embeddings for these text chunks using OpenAI's embedding model
#     3. Store these embeddings in Neo4j with the proper index configuration
    
#     Check the sidebar for more detailed setup instructions.
#     """)

st.markdown("---")
st.markdown("Built with Streamlit, LangChain, OpenAI, and Neo4j")