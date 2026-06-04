"""
based on https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/#2-advanced-capability-2-text-to-sql-with-query-time-row-retrieval-along-with-table-retrieval

"""

import json
import os
from pathlib import Path
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.retrievers import SQLRetriever
from llama_index.core.query_pipeline import FnComponent
from typing import List
import mlflow.llama_index
from pydantic import BaseModel, Field
from llama_index.llms.ollama import Ollama
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    InputComponent,
)
from llama_index.core.llms import ChatResponse
from llama_index.core import SQLDatabase, VectorStoreIndex, PromptTemplate
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from sqlalchemy import create_engine
import mlflow

DB_HOST: str = "localhost" # SQL host
DB_PORT: str = "3307" # SQL port
DB_USER: str = "user" # SQL user
DB_PASSWORD: str = "password" # SQL password 
DB_DATABASE: str = "cyberchase" # Database name
DB_TABLES: List[str] = ["episodes"] # Tables to use (NOT USED)
OLLAMA_HOST: str = "localhost:11434" # Ollama host
TEXT_TO_SQL_MODEL: str = "mistral" # LLM model name
BG_MODEL : str = "hf.co/INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0-GGUF:Q4_K_S" # LLM model for generation in bulgarian

MLFLOW_URI = "http://localhost:5000"

BG = True # If the bulgarian pipeline should be used or not



"mlflow ui --port 5000"
"mlflow gc --tracking-uri http://localhost:5000"

# Make a MLFlow experiment and choose a file to read questions from
base_dir = Path(__file__).parent
if(BG == False): 
    mlflow.set_experiment(f"{DB_DATABASE} - {TEXT_TO_SQL_MODEL}")
    file_path = base_dir / 'DatabasesQA.json'
else:
    mlflow.set_experiment(f"(BG) {DB_DATABASE} - {TEXT_TO_SQL_MODEL}")
    file_path = base_dir / 'DatabasesQA_BG.json'
# Connect to mlflow server
mlflow.set_tracking_uri(
    MLFLOW_URI
)
# Log the activity of the pipeline
mlflow.llama_index.autolog()
# Load the questions and answers from the JSON file
with file_path.open('r', encoding="utf-8") as file:
    data = json.load(file)


# Callback manager for debugging and measuring the performance of the pipeline
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

# Contains name and summary of a table
class TableInfo(BaseModel): 
    table_name: str
    table_summary: str 

# Class for table summary that incudes the summary of a table
class TableSummary(BaseModel): 
    table_summary: str = Field(..., description="short, concise summary/caption of the table")

# Connect to database
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}")
# SQLAlchemy engine wrapper
sql_database = SQLDatabase(engine)

# Connect to Ollama LLM model
llm = Ollama(model=TEXT_TO_SQL_MODEL, base_url=OLLAMA_HOST, request_timeout=180.0, context_window=30000)
bg_llm = Ollama(model=BG_MODEL, base_url=OLLAMA_HOST, request_timeout=180.0, context_window=5000)

# Prompt for generating table summary
prompt_str = """\
Give me a summary of the {table_name} table with the following JSON format.

Example output:
{
  "table_summary": "(summary here)"
}

Table:
{table_str}

Summary: """

# Generate structured output for TableSummary class from the LLM
program = LLMTextCompletionProgram.from_defaults(
    output_cls=TableSummary,
    llm=llm,
    prompt_template_str=prompt_str,
    verbose=True
)

db_names = ""
# Make a MLFlow run for the table summary generation
with mlflow.start_run(run_name="Table summary generation"):
    # Generate summary for all usable tables
    table_infos = []
    for t in sql_database.get_usable_table_names():
        # Get table and cloumn names and add them to the db_names string
        db_names += t + ", "
        columns = sql_database.get_table_columns(t)
        column_names = [col["name"] for col in columns]
        for col in column_names:
            db_names += col + ", "
        # Get table schema and sample rows
        table_schema = sql_database.run_sql(f"DESCRIBE {t}") 
        table_sample_rows = sql_database.run_sql(f"SELECT * FROM {t} LIMIT 4")

        table_str = f"\{table_schema}\n\nExample rows of table:\n{table_sample_rows}"

        # Generate TableSummary object
        table_summary = program(table_str=table_str, table_name=t)

        table_infos.append(TableInfo(table_name=t, table_summary=table_summary.table_summary))
    pass
            
print(table_infos)
# Propmpt to generate transaltion of the query in english
bg_prompt = """\
Give me a english translation of the following question in bulgarian with the following JSON format.

Those words might be useful during translation: {db_names}

Example output:
{
  "en_translation": "(translation here)"
}

Bulgarian question:
{bg_question}

English translation:
"""
# Class to store the translation
class ENTranslation(BaseModel): 
    en_translation: str = Field(..., description="translation of the question in english")
# Generator for the translation
bg_program = LLMTextCompletionProgram.from_defaults(
        output_cls=ENTranslation,
        llm=bg_llm,
        prompt_template_str=bg_prompt,
        verbose=True
    )

# Translate query from bulgarian to english
def translateToEN(query: str):
    translation = bg_program(bg_question=query, db_names=db_names)
    return translation.en_translation

translate_component = FnComponent(fn=translateToEN)



# Generate llama_index node for each table in the database
table_node_mapping = SQLTableNodeMapping(sql_database)
# Create SQLTableSchema object containing the name and summary of each table
table_schema_objs = [SQLTableSchema(table_name=t.table_name, context_str=t.table_summary) for t in table_infos]

# Embed the table infos into vectors, map them to the tables and store them in an index
embed_model = OllamaEmbedding(model_name=TEXT_TO_SQL_MODEL)
obj_index = ObjectIndex.from_objects(table_schema_objs, table_node_mapping, VectorStoreIndex, embed_model=embed_model)

# Retreiver for the most semantically similiar tables to the query
obj_retriever = obj_index.as_retriever(similarity_top_k=4)

# SQL query runner
sql_retriever = SQLRetriever(sql_database)

# Get the table description and context for each table in the input 
def get_tables_context_str(table_schema_objs: List[SQLTableSchema]):
    context_strs=[]
    for table_schema_obj in table_schema_objs:
        # String containing the schema and foreign keys of a table
        table_info = sql_database.get_single_table_info(
            table_schema_obj.table_name
        )
        if table_schema_obj.context_str:
            table_opt_context = "Table description is: "
            table_opt_context += table_schema_obj.context_str
            table_info += table_opt_context

        context_strs.append(table_info)
    return "\n\n".join(context_strs)

# Make component for the pipeline of the function get_tables_context_str
table_parser_component = FnComponent(fn=get_tables_context_str)

# Parse the output of the Text2SQL LLM and return the SQL query generated by the model
def parse_response_sql(response: ChatResponse) -> str:
    # Extract the SQL query generated by the LLM
    response = response.message.content
    sql_query_start = response.find("SQLQuery:")
    if sql_query_start != -1:
        response = response[sql_query_start:]
        if response.startswith("SQLQuery:"):
            response = response[len("SQLQuery:") :]
    sql_result_start = response.find("SQLResult:")
    if sql_result_start != -1:
        response = response[:sql_result_start]

    # Get the raw text of the SQL query
    sql_query = response.strip().strip("```").strip()
    if sql_query.lower().startswith("sql"):
        sql_query = sql_query.removeprefix("sql").lstrip()

    sql_query = sql_query.split(';',1)[0]
    return sql_query

# Make component for the pipeline of the function parse_response_sql
sql_parser_comp = FnComponent(fn=parse_response_sql)

# Text2SQL prompt
text2sql_prompt_str = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for a few relevant columns given the question. Prioritiese using LIKE %(string)% instead of = when searching for strings.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Pay attention to which column is in which table. Also, qualify column names with the table name when needed. You are required to use the following format, each taking one line:

Question: (Question here)
SQLQuery: (SQL Query to run)
SQLResult: (Result of the SQLQuery)

Only use tables listed below.
{schema}

Question: {query_str}
SQLQuery: """


# Create PromptTemplate for the pipeline
text2sql_prompt = PromptTemplate(text2sql_prompt_str, dialect=engine.dialect.name)

# Define query pipeline
if(BG == True):
    response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results in bulgarian.\n"
    "Query: {query_str}\n"
    "SQL: {sql_query}\n"
    "SQL Response: {context_str}\n"
    "Response: "
    )
    response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)

    qp = QP(
        modules={
            "BG input" : InputComponent(),
            "User query": translate_component,
            "Table retriever": obj_retriever,
            "Table output parser": table_parser_component,
            "Text2SQL prompt": text2sql_prompt,
            "Text2SQL LLM": llm,
            "SQL output parser": sql_parser_comp,
            "SQL retreiver": sql_retriever,
            "Final response prompt": response_synthesis_prompt,
            "Final response LLM": bg_llm,
        },
        verbose=False,
        callback_manager=callback_manager
    )
    qp.add_chain(["BG input","User query", "Table retriever","Table output parser"]) # Get the description of the semantically most similiar tables to the user query
    qp.add_link("BG input", "Final response prompt", dest_key="query_str") # Give the user query to the Final response prompt
    qp.add_chain(["User query", "Table retriever","Table output parser"]) # Get the description of the semantically most similiar tables to the user query
    qp.add_link("User query", "Final response prompt", dest_key="query_str") # Give the user query to the Final response prompt
    qp.add_link("User query","Text2SQL prompt",dest_key="query_str") # Give the user query to the Text2SQL prompt
    qp.add_link("Table output parser","Text2SQL prompt",dest_key="schema") # Give the descriptions of the tables to the Text2SQL prompt
    qp.add_chain(["Text2SQL prompt", "Text2SQL LLM", "SQL output parser", "SQL retreiver"]) # Generate LLM output using the Text2SQL prompt, extract the SQL query and run it in the database
    qp.add_link("SQL output parser", "Final response prompt", dest_key="sql_query") # Give the generated SQL query to the Final response prompt 
    qp.add_link(
        "SQL retreiver", "Final response prompt", dest_key="context_str"
    ) # Give the results from the SQL query to the Final response prompt

    qp.add_link("Final response prompt", "Final response LLM") # Generate the Final resoponse
else: 
    # Final response prompt
    response_synthesis_prompt_str = (
        "Given an input question, synthesize a response from the query results.\n"
        "Query: {query_str}\n"
        "SQL: {sql_query}\n"
        "SQL Response: {context_str}\n"
        "Response: "
    )
    response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)

    qp = QP(
    modules={
        "User query": InputComponent(),
        "Table retriever": obj_retriever,
        "Table output parser": table_parser_component,
        "Text2SQL prompt": text2sql_prompt,
        "Text2SQL LLM": llm,
        "SQL output parser": sql_parser_comp,
        "SQL retreiver": sql_retriever,
        "Final response prompt": response_synthesis_prompt,
        "Final response LLM": llm,
    },
    verbose=False,
    callback_manager=callback_manager
    )
    qp.add_chain(["User query", "Table retriever","Table output parser"]) # Get the description of the semantically most similiar tables to the user query
    qp.add_link("User query", "Final response prompt", dest_key="query_str") # Give the user query to the Final response prompt
    qp.add_link("User query","Text2SQL prompt",dest_key="query_str") # Give the user query to the Text2SQL prompt
    qp.add_link("Table output parser","Text2SQL prompt",dest_key="schema") # Give the descriptions of the tables to the Text2SQL prompt
    qp.add_chain(["Text2SQL prompt", "Text2SQL LLM", "SQL output parser", "SQL retreiver"]) # Generate LLM output using the Text2SQL prompt, extract the SQL query and run it in the database
    qp.add_link("SQL output parser", "Final response prompt", dest_key="sql_query") # Give the generated SQL query to the Final response prompt 
    qp.add_link(
        "SQL retreiver", "Final response prompt", dest_key="context_str"
    ) # Give the results from the SQL query to the Final response prompt

    qp.add_link("Final response prompt", "Final response LLM") # Generate the Final resoponse
    



# Run the pipeline for all questions for the database
for q in data[DB_DATABASE]["questions"]:
    # Make MLFlow run with the name of the question
    with mlflow.start_run(run_name=q["question"]):
        # Log the question
        mlflow.log_param("Question", q["question"])
        try: 
            # Run the pipeline
            response,intermidiates = qp.run_with_intermediates(query=q["question"])
           
            if(BG == True):
                # Log the translation
                mlflow.log_param("Question EN", intermidiates["User query"].outputs["output"])
                
        except Exception as e:
            # q["answer"] = "Failed"
            # Log if there is an error during runtime
            print(f"Pipeline error: {e}")
            mlflow.log_param("SQL_Output", "Failed")
            mlflow.set_tag("Result", "Runtime error")
            continue
        
        sql_query = intermidiates["SQL output parser"].outputs["output"]
        sql_output = intermidiates["Final response prompt"].inputs["context_str"][0].text
        print (f"{q['question']}: {sql_query} - {sql_output} \n")

        q["example_query"] = sql_query
        q["answer"] = sql_output
        # Log the generated SQL query and results and the true answer from the QA file
        mlflow.log_param("SQL_Query", sql_query)
        mlflow.log_param("SQL_Output", sql_output)
        mlflow.log_param("True answer", q["true_answer"])
        # Log if the SQL result is the same as the one in the QA 
        if (sql_output == q["true_answer"]): 
           mlflow.set_tag("Result", "Success")
        else:
           mlflow.set_tag("Result", "Wrong SQL output")
        pass
    



print('Done')

