from typing import Any
import streamlit as st
import pandas as pd
import mysql.connector
import google.generativeai as genai
from google.generativeai import types
import os
from dotenv import load_dotenv

MODEL = "gemini-2.0-flash"
TEMPERATURE = 0.7
SYSTEM_PROMPT = """You are an SQL expert. Given a natural language question, you will return the SQL query that answers it\n
When searching for names, use the LOWER() function to perform case-insensitive matching. For example, use WHERE LOWER(column_name) = LOWER('search_term') instead of ILIKE.\n
When creating or updating a row, use RETURNING * to return the updated/saved information."""

# --- Start the Google Gemini API client ---
api = "AIzaSyD8FrLR6bmeG6NFEztRtvdfoPH-FGIkaHc"
genai.configure(api_key=api)

model = genai.GenerativeModel('gemini-1.5-flash')

# ---- Create a demo database ----
def setup_demo_database() -> mysql.connector.connection.MySQLConnection:
    # DB connection
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Aftab@786"
    )
    cursor = conn.cursor()

    # Create database if not exists
    cursor.execute("CREATE DATABASE IF NOT EXISTS product_db1")
    cursor.execute("USE product_db1")

    # Drop existing tables if they exist (to avoid foreign key issues)
    cursor.execute("DROP TABLE IF EXISTS orders")
    cursor.execute("DROP TABLE IF EXISTS client")

    # Generate DB tables
    cursor.execute('''
    CREATE TABLE client (
        id INT PRIMARY KEY AUTO_INCREMENT,
        name VARCHAR(255),
        email VARCHAR(255),
        created_at DATE
    )
    ''')

    cursor.execute('''
    CREATE TABLE orders (
        id INT PRIMARY KEY AUTO_INCREMENT,
        client_id INT,
        product VARCHAR(255),
        price DECIMAL(10,2),
        created_at DATE,
        FOREIGN KEY (client_id) REFERENCES client(id)
    )
    ''')

    # Check if data exists before inserting
    cursor.execute("SELECT COUNT(*) FROM client")
    if cursor.fetchone()[0] == 0:
        # Seed data
        cursor.executemany("INSERT INTO client (name, email, created_at) VALUES (%s, %s, %s)", [
            ("Harry Potter", "harry@email.com", "2024-01-10"),
            ("Hermione Granger", "hermione@email.com", "2024-02-15"),
            ("Draco Malfoy", "draco@email.com", "2024-03-20"),
            ("Lord Voldemort", "voldemort@email.com", "2024-04-05"),
            ("Ron Weasley", "ron@email.com", "2024-05-12")
        ])

        cursor.executemany("INSERT INTO orders (client_id, product, price, created_at) VALUES (%s, %s, %s, %s)", [
            (1, "Notebook", 3500.00, "2024-06-10"),
            (2, "Smartphone", 2100.00, "2024-06-15"),
            (3, "Monitor", 1200.00, "2024-07-05"),
            (1, "Keyboard", 150.00, "2024-07-10"),
            (2, "Mouse", 80.00, "2024-07-15"),
            (4, "Printer", 950.00, "2024-08-01"),
            (5, "Headphone", 300.00, "2024-08-05"),
            (3, "Webcam", 220.00, "2024-08-10"),
            (1, "SSD 1TB", 450.00, "2024-09-01"),
            (4, "Router", 180.00, "2024-09-10"),
            (1, "Notebook", 3500.00, "2025-01-10"),
        ])

    conn.commit()
    return conn

# ---- Extract database schema ----
def get_database_schema(conn: mysql.connector.connection.MySQLConnection) -> dict:
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()

    schema = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"DESCRIBE {table_name}")
        columns = cursor.fetchall()
        schema[table_name] = [{"name": col[0], "type": col[1]} for col in columns]

    return schema

# ---- Generate SQL query from text prompt ----
def generate_sql_query(text_prompt: str, schema: Any) -> str:
    # Generate the schema context
    schema_context = "Database schema:\n"
    for table, columns in schema.items():
        schema_context += f"Tables: {table}\n"
        for col in columns:
            schema_context += f"- {col['name']} ({col['type']})\n"

    # Build the full prompt
    full_prompt = f"""
    {schema_context}
    {st.session_state.get("system_prompt", SYSTEM_PROMPT)}
    Based on the database schema above, generate a valid SQL query for the following question:
    "{text_prompt}"
    Return only the SQL code, without additional explanations.
    """

    # Call Gemini API
    response = model.generate_content(
       contents=full_prompt,
       generation_config=types.GenerationConfig(
           temperature=st.session_state.get("temperature_slider", 0.7)
           ),
    )

    # Extract the SQL query from response
    sql_query = response.text.strip()

    # Remove markdown code if any
    if sql_query.startswith("```sql"):
        sql_query = sql_query[6:]
    if sql_query.endswith("```"):
        sql_query = sql_query[:-3]

    return sql_query.strip()

# ---- Execute query ----
def execute_query(conn: mysql.connector.connection.MySQLConnection, query: str) -> (tuple[pd.DataFrame, None] | tuple[None, str]):
    try:
        cursor = conn.cursor()
        query_lower = query.strip().lower()

        if query_lower.startswith("select"):
            result = pd.read_sql_query(query, conn)
            return result, None

        elif "returning" in query_lower:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            conn.commit()
            return pd.DataFrame(rows, columns=columns), None

        else:
            cursor.execute(query)
            conn.commit()
            return pd.DataFrame(), None

    except Exception as e:
        return None, str(e)

# ---- Explain the query result ----
def explain_results(query: str, results: pd.DataFrame, user_prompt: str) -> str:
    if results is None or results.empty:
        return "No results found for the given query."
    
    # Get the first 10 results from dataframe
    results_sample = results.head(10).to_string()
    
    full_prompt = f"""
    The following query was executed:
    {query}
    
    For question: "{user_prompt}"
    
    Results:
    {results_sample}
    
    Please explain these results clearly and concisely in English.
    If there are many lines, mention only the most important insights or patterns observed.
    """

    response = model.generate_content(
        contents=full_prompt,
        generation_config=types.GenerationConfig(
            temperature=st.session_state.get("temperature_slider", 0.7)
            ),
        )
    
    return response.text

# ---- Initialize DB ----
def get_database_connection() -> mysql.connector.connection.MySQLConnection:
    return setup_demo_database()

# ---- Process query ----
def process_query(prompt, schema, db_conn) -> None:
    # Show assistant message
    with st.chat_message("assistant"):
        # Result container
        sql_container = st.empty()
        results_container = st.empty()
        explanation_container = st.empty()
        
        # Generate the SQL container
        sql_container.text("Generating SQL query...")
        sql_query = generate_sql_query(prompt, schema)
        sql_container.code(sql_query, language="sql")
        
        # Execute the query
        results_container.text("Executing query...")
        results, error = execute_query(db_conn, sql_query)
      
        if error:
            results_container.error(f"Query error: {error}")
            explanation = None
        else:
            # Show results
            results_container.subheader("Results:")
            results_container.dataframe(results)
          
            # Show explanation
            explanation_container.text("Analysing results...")
            explanation = explain_results(sql_query, results, prompt)
            explanation_container.subheader("Explanation:")
            explanation_container.write(explanation)
  
    # Add response to history dictionary
    st.session_state.messages.append({
        "role": "assistant",
        "content": {
            "sql": sql_query,
            "error": error,
            "results": results if error is None else None,
            "explanation": explanation
        }
    })

def main() -> None:
    # ---- Page config -----
    st.set_page_config(
        page_title="Text to SQL",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Get DB connection
    db_conn = get_database_connection()
    
    # Get DB schema
    schema = get_database_schema(db_conn)

    # ---- Sidebar config -----
    with st.sidebar:
        st.header("Configuration")
        st.slider("LLM Temperature", 0.0, 1.0, 0.7, 0.1, key="temperature_slider")
        st.text_area("System prompt", SYSTEM_PROMPT, height=350, key="system_prompt")
        st.divider()

        # Schemas
        st.header("📝 Database schemas")
        for table, columns in schema.items():
            with st.expander(f"Table: {table}"):
                cols_df = pd.DataFrame(columns)
                st.dataframe(cols_df)

    st.title("🤖 Text to SQL agent")

    # ---- Initialize message state ----
    if "messages" not in st.session_state:
        st.session_state.messages = []
  
    # Show past messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                # Show assistant query, results and explanation
                st.code(message["content"]["sql"], language="sql")
                
                if message["content"]["error"]:
                    st.error(f"Query error: {message['content']['error']}")
                else:
                    st.subheader("Results:")
                    st.dataframe(message["content"]["results"])
                    
                    st.subheader("Explanation:")
                    st.write(message["content"]["explanation"])
  
    # User input
    if prompt := st.chat_input("Ask a question about the data..."):
        # Add user message to message state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Show user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process user prompt
        process_query(prompt, schema, db_conn)

if __name__ == "__main__":
    main()
  