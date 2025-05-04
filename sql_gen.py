"""Example demonstrating how to use PydanticAI to generate SQL queries based on user input.

Run postgres with:

    mkdir postgres-data
    docker run --rm -e POSTGRES_PASSWORD=postgres -p 54320:5432 postgres

Run with:

    uv run -m pydantic_ai_examples.sql_gen "show me logs from yesterday, with level 'error'"
"""

import asyncio
import sys
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, Union

import asyncpg
import logfire
from annotated_types import MinLen
from devtools import debug
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.providers.google_gla import GoogleGLAProvider

# Load environment variables from .env file
load_dotenv()

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_asyncpg()

DB_SCHEMA = """
CREATE TABLE records (
    created_at timestamptz,
    start_timestamp timestamptz,
    end_timestamp timestamptz,
    trace_id text,
    span_id text,
    parent_span_id text,
    level log_level,
    span_name text,
    message text,
    attributes_json_schema text,
    attributes jsonb,
    tags text[],
    is_exception boolean,
    otel_status_message text,
    service_name text
);
"""

@dataclass
class Deps:
    conn: asyncpg.Connection

# Define tools before initializing the agent

async def select_records(
    ctx: RunContext[Deps], 
    columns: list[str] | None = None,
    filters: str | None = None, 
    limit: int = 100,
) -> list[dict[str, Any]] | str:
    """Select records from the 'records' table based on specified criteria.

    Args:
        columns: List of columns to select. Defaults to all columns (*).
        filters: SQL WHERE clause conditions (e.g., "level = 'error' AND start_timestamp::date > CURRENT_DATE - INTERVAL '1 day'").
        limit: Maximum number of records to return.
    """
    conn = ctx.deps.conn
    select_cols = ", ".join(f'"{col}"' for col in columns) if columns else "*"
    query = f"SELECT {select_cols} FROM records"
    if filters:
        query += f" WHERE {filters}"
    query += f" LIMIT {limit}"

    logfire.info(f"Executing tool-generated query: {query}")
    try:
        rows = await conn.fetch(query)
        results = [dict(row) for row in rows]
        
        # If no results, fetch distinct values like before (simplified)
        if not results and filters:
             # Basic attempt to find columns in filters
            potential_cols = []
            all_db_cols = await conn.fetch("SELECT column_name FROM information_schema.columns WHERE table_name = 'records'")
            all_db_col_names = [r['column_name'] for r in all_db_cols]
            for col in all_db_col_names:
                if col in filters:
                     potential_cols.append(col)
            if not potential_cols:
                 potential_cols = [c for c in ['level', 'service_name', 'tags'] if c in all_db_col_names] # Fallback

            distinct_info = "\n\n**No results found. Available values for potentially relevant columns:**\n"
            for col in potential_cols[:3]: # Limit distinct value checks
                try:
                    distinct_vals = await conn.fetch(f'SELECT DISTINCT "{col}" FROM records LIMIT 10')
                    if distinct_vals:
                        vals = [dict(r)[col] for r in distinct_vals]
                        # Handle list of lists for tags
                        if col == 'tags' and vals and isinstance(vals[0], list):
                            flat_tags = {tag for sublist in vals if sublist for tag in sublist}
                            distinct_info += f"- **{col}**: {', '.join(repr(v) for v in flat_tags)}\n"
                        else:
                            distinct_info += f"- **{col}**: {', '.join(repr(v) for v in vals)}\n"
                except Exception as e:
                    logfire.warn(f"Could not fetch distinct values for {col}: {e}")
            return distinct_info
        
        # Return results, potentially formatting them nicely
        if results:
            return f"Found {len(results)} records:\n" + "\n".join(str(r) for r in results[:5]) + ("\n..." if len(results) > 5 else "")
        else:
             return "No records found matching your criteria."

    except asyncpg.exceptions.PostgresError as e:
        logfire.error(f"Error executing tool query: {e}")
        # Inform the LLM about the error so it can potentially fix the filter
        raise ModelRetry(f'Error executing SQL: {e}. Use the exact column names from the schema and correct SQL syntax for filters.') from e

# Get API key from environment
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# Explicitly create the provider with the API key
provider = GoogleGLAProvider(api_key=gemini_api_key)

agent: Agent[Deps, str] = Agent(
    model='google-gla:gemini-1.5-flash',
    provider=provider,
    output_type=str,  # Output is now a string summary/result
    instrument=True,
)


@agent.tool
async def select_records(
    ctx: RunContext[Deps], 
    columns: list[str] | None = None,
    filters: str | None = None, 
    limit: int = 100,
) -> list[dict[str, Any]] | str:
    """Select records from the 'records' table based on specified criteria.

    Args:
        columns: List of columns to select. Defaults to all columns (*).
        filters: SQL WHERE clause conditions (e.g., "level = 'error' AND start_timestamp::date > CURRENT_DATE - INTERVAL '1 day'").
        limit: Maximum number of records to return.
    """
    conn = ctx.deps.conn
    select_cols = ", ".join(f'"{col}"' for col in columns) if columns else "*"
    query = f"SELECT {select_cols} FROM records"
    if filters:
        query += f" WHERE {filters}"
    query += f" LIMIT {limit}"

    logfire.info(f"Executing tool-generated query: {query}")
    try:
        rows = await conn.fetch(query)
        results = [dict(row) for row in rows]
        
        # If no results, fetch distinct values like before (simplified)
        if not results and filters:
             # Basic attempt to find columns in filters
            potential_cols = []
            all_db_cols = await conn.fetch("SELECT column_name FROM information_schema.columns WHERE table_name = 'records'")
            all_db_col_names = [r['column_name'] for r in all_db_cols]
            for col in all_db_col_names:
                if col in filters:
                     potential_cols.append(col)
            if not potential_cols:
                 potential_cols = [c for c in ['level', 'service_name', 'tags'] if c in all_db_col_names] # Fallback

            distinct_info = "\n\n**No results found. Available values for potentially relevant columns:**\n"
            for col in potential_cols[:3]: # Limit distinct value checks
                try:
                    distinct_vals = await conn.fetch(f'SELECT DISTINCT "{col}" FROM records LIMIT 10')
                    if distinct_vals:
                        vals = [dict(r)[col] for r in distinct_vals]
                        # Handle list of lists for tags
                        if col == 'tags' and vals and isinstance(vals[0], list):
                            flat_tags = {tag for sublist in vals if sublist for tag in sublist}
                            distinct_info += f"- **{col}**: {', '.join(repr(v) for v in flat_tags)}\n"
                        else:
                            distinct_info += f"- **{col}**: {', '.join(repr(v) for v in vals)}\n"
                except Exception as e:
                    logfire.warn(f"Could not fetch distinct values for {col}: {e}")
            return distinct_info
        
        # Return results, potentially formatting them nicely
        if results:
            return f"Found {len(results)} records:\n" + "\n".join(str(r) for r in results[:5]) + ("\n..." if len(results) > 5 else "")
        else:
             return "No records found matching your criteria."

    except asyncpg.exceptions.PostgresError as e:
        logfire.error(f"Error executing tool query: {e}")
        # Inform the LLM about the error so it can potentially fix the filter
        raise ModelRetry(f'Error executing SQL: {e}. Use the exact column names from the schema and correct SQL syntax for filters.') from e


# New system prompt instructing the use of tools
@agent.system_prompt
async def system_prompt() -> str:
    return f"""\
You are an assistant that helps query a PostgreSQL database containing log records.
Use the available tools to answer the user's request about the data.
Do NOT generate raw SQL queries yourself. Use the 'select_records' tool.

Database schema for the 'records' table:

{DB_SCHEMA}

today's date = {date.today()}

When using the 'select_records' tool:
- Provide the SQL `WHERE` clause logic in the `filters` argument.
- Ensure filter syntax is correct PostgreSQL.
- Use column names exactly as defined in the schema.
- You can specify `columns` to retrieve only specific fields.
- The tool handles query execution and returns results or relevant information if no results are found.
"""


async def main():
    if len(sys.argv) == 1:
        prompt = 'show me logs from yesterday, with level "error"'
    else:
        prompt = sys.argv[1]

    db_dsn = 'postgresql://postgres:postgres@localhost:5432' # Correct port
    db_name = 'postgres'

    async with database_connect(
        db_dsn, db_name
    ) as conn:
        deps = Deps(conn)
        # The agent now handles tool execution internally based on the prompt
        result = await agent.run(prompt, deps=deps)

        # The result.output is now the final string response from the agent
        # (either summarizing tool execution or the direct string output of the tool)
        print("\n--- Agent Response ---")
        if isinstance(result.output, str):
            print(result.output)
        else:
            # In case the tool returned raw data (though we aimed for string)
            debug(result.output)
        print("--------------------")

    # Old logic for executing query and fetching distinct values is removed,
    # as it's now handled within the select_records tool.


# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
@asynccontextmanager
async def database_connect(server_dsn: str, database: str) -> AsyncGenerator[Any, None]:
    with logfire.span('check and create DB'):
        conn = await asyncpg.connect(server_dsn) # Connect to the server DSN first
        try:
            db_exists = await conn.fetchval(
                'SELECT 1 FROM pg_database WHERE datname = $1', database
            )
            if not db_exists:
                await conn.execute(f'CREATE DATABASE {database}')
        finally:
            await conn.close()

    # Now connect to the specific database using the full DSN
    db_full_dsn = f'{server_dsn}/{database}'
    conn = await asyncpg.connect(db_full_dsn)
    try:
        with logfire.span('create schema'):
            async with conn.transaction():
                # Check if the enum type exists before creating
                type_exists = await conn.fetchval("SELECT 1 FROM pg_type WHERE typname = 'log_level'")
                if not type_exists:
                     await conn.execute(
                        "CREATE TYPE log_level AS ENUM ('debug', 'info', 'warning', 'error', 'critical')"
                     )
                # Check if the table exists before creating
                table_exists = await conn.fetchval(
                    "SELECT 1 FROM information_schema.tables WHERE table_name = 'records'"
                )
                if not table_exists:
                    await conn.execute(DB_SCHEMA)
        yield conn
    finally:
        await conn.close()


if __name__ == '__main__':
    asyncio.run(main())
