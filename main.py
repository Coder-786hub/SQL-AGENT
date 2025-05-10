from typing import Any
import streamlit as st
import pandas as pd
import mysql.connector
import google.generativeai as genai
from google.generativeai import types

# === Google Gemini config ===
MODEL = "gemini-2.0-flash"
TEMPERATURE = 0.7
SYSTEM_PROMPT = """You are an SQL expert. Given a natural language question, you will return the SQL query that answers it.
Use LOWER() for case-insensitive string matching. Use RETURNING * when inserting/updating to return results."""

genai.configure(api_key="AIzaSyD8FrLR6bmeG6NFEztRtvdfoPH-FGIkaHc")
model = genai.GenerativeModel(MODEL)


# === Step 1: Get DB connection from user ===
def get_user_database_connection():
    st.subheader("🔐 Enter MySQL Database Connection Details")
    if "db_conn" not in st.session_state:
        st.session_state.db_conn = None

    host = st.text_input("Host", value="localhost")
    port = st.number_input("Port", value=3306)
    user = st.text_input("Username")
    password = st.text_input("Password", type="password")
    database = st.text_input("Database Name")

    if st.button("Connect to Database"):
        try:
            conn = mysql.connector.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database
            )
            if conn.is_connected():
                st.session_state.db_conn = conn
                st.success("✅ Connected to the database!")
        except mysql.connector.Error as e:
            st.error(f"❌ Connection failed: {e}")
            st.session_state.db_conn = None


# === Step 2: Get schema ===
def get_database_schema(conn) -> dict:
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()

    schema = {}
    for (table,) in tables:
        cursor.execute(f"DESCRIBE {table}")
        columns = cursor.fetchall()
        schema[table] = [{"name": col[0], "type": col[1]} for col in columns]
    return schema


# === Step 3: Generate SQL ===
def generate_sql_query(text_prompt: str, schema: Any) -> str:
    schema_context = "Database schema:\n"
    for table, columns in schema.items():
        schema_context += f"Table: {table}\n"
        for col in columns:
            schema_context += f"- {col['name']} ({col['type']})\n"

    full_prompt = f"""
    {schema_context}
    {st.session_state.get("system_prompt", SYSTEM_PROMPT)}
    Generate a SQL query for the question:
    "{text_prompt}"
    Return only SQL code, no explanations.
    """

    response = model.generate_content(
        contents=full_prompt,
        generation_config=types.GenerationConfig(
            temperature=st.session_state.get("temperature_slider", 0.7)
        ),
    )

    sql_query = response.text.strip()
    return sql_query.replace("```sql", "").replace("```", "").strip()


# === Step 4: Execute SQL ===
def execute_query(conn, query):
    try:
        cursor = conn.cursor()
        if query.lower().startswith("select"):
            result = pd.read_sql_query(query, conn)
            return result, None
        elif "returning" in query.lower():
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            conn.commit()
            return pd.DataFrame(rows, columns=columns), None
        else:
            cursor.execute(query)
            conn.commit()
            return pd.DataFrame(), None
    except Exception as e:
        return None, str(e)


# === Step 5: Explain results ===
def explain_results(query, results, user_prompt):
    if results is None or results.empty:
        return "No results found."

    result_sample = results.head(10).to_string()
    prompt = f"""
    SQL query executed:
    {query}
    Based on the question: "{user_prompt}"
    Sample results:
    {result_sample}
    Explain this output briefly.
    """
    response = model.generate_content(
        contents=prompt,
        generation_config=types.GenerationConfig(
            temperature=st.session_state.get("temperature_slider", 0.7)
        )
    )
    return response.text


# === Step 6: Process user question ===
def process_query(prompt, schema, conn):
    with st.chat_message("assistant"):
        sql_placeholder = st.empty()
        result_placeholder = st.empty()
        explain_placeholder = st.empty()

        sql_placeholder.text("Generating SQL...")
        sql = generate_sql_query(prompt, schema)
        sql_placeholder.code(sql, language="sql")

        result_placeholder.text("Running query...")
        results, error = execute_query(conn, sql)
        if error:
            result_placeholder.error(f"Query Error: {error}")
            explanation = None
        else:
            result_placeholder.subheader("Results:")
            result_placeholder.dataframe(results)
            explain_placeholder.text("Explaining results...")
            explanation = explain_results(sql, results, prompt)
            explain_placeholder.subheader("Explanation:")
            explain_placeholder.write(explanation)

        st.session_state.messages.append({
            "role": "assistant",
            "content": {
                "sql": sql,
                "error": error,
                "results": results,
                "explanation": explanation
            }
        })


# === Main App ===
def main():
    st.set_page_config("Text to SQL Gemini App", "🤖", layout="centered")

    st.title("🤖 Text to SQL Agent using Gemini + MySQL")

    # Sidebar setup
    with st.sidebar:
        st.header("🔧 Settings")
        st.slider("LLM Temperature", 0.0, 1.0, 0.7, 0.1, key="temperature_slider")
        st.text_area("System Prompt", SYSTEM_PROMPT, height=200, key="system_prompt")

    # Connection
    get_user_database_connection()

    if st.session_state.get("db_conn"):
        conn = st.session_state.db_conn
        schema = get_database_schema(conn)

        # Show schema in sidebar
        with st.sidebar:
            st.header("🧱 DB Schema")
            for table, columns in schema.items():
                with st.expander(f"📄 {table}"):
                    st.dataframe(pd.DataFrame(columns))

        # Message state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.write(msg["content"])
                else:
                    st.code(msg["content"]["sql"], language="sql")
                    if msg["content"]["error"]:
                        st.error(msg["content"]["error"])
                    else:
                        st.subheader("Results:")
                        st.dataframe(msg["content"]["results"])
                        st.subheader("Explanation:")
                        st.write(msg["content"]["explanation"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your data..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            process_query(prompt, schema, conn)


if __name__ == "__main__":
    main()
