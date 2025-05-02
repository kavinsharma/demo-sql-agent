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
from typing_extensions import TypeAlias
from dotenv import load_dotenv

from pydantic_ai import Agent, ModelRetry, RunContext, format_as_xml
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
SQL_EXAMPLES = [
    {
        'request': 'show me records where foobar is false',
        'response': "SELECT * FROM records WHERE attributes->>'foobar' = false",
    },
    {
        'request': 'show me records where attributes include the key "foobar"',
        'response': "SELECT * FROM records WHERE attributes ? 'foobar'",
    },
    {
        'request': 'show me records from yesterday',
        'response': "SELECT * FROM records WHERE start_timestamp::date > CURRENT_TIMESTAMP - INTERVAL '1 day'",
    },
    {
        'request': 'show me error records with the tag "foobar"',
        'response': "SELECT * FROM records WHERE level = 'error' and 'foobar' = ANY(tags)",
    },
]


@dataclass
class Deps:
    conn: asyncpg.Connection


class Success(BaseModel):
    """Response when SQL could be successfully generated."""

    sql_query: Annotated[str, MinLen(1)]
    explanation: str = Field(
        '', description='Explanation of the SQL query, as markdown'
    )
    results: list[dict[str, Any]] | None = Field(
        None, description='Results of executing the SQL query'
    )


class InvalidRequest(BaseModel):
    """Response the user input didn't include enough information to generate SQL."""

    error_message: str


Response: TypeAlias = Union[Success, InvalidRequest]

# Get API key from environment
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# Explicitly create the provider with the API key
provider = GoogleGLAProvider(api_key=gemini_api_key)

agent: Agent[Deps, Response] = Agent(
    model='google-gla:gemini-1.5-flash',
    provider=provider,
    # Type ignore while we wait for PEP-0747, nonetheless unions will work fine everywhere else
    output_type=Response,  # type: ignore
    deps_type=Deps,
    instrument=True,
)


@agent.system_prompt
async def system_prompt() -> str:
    return f"""\
Given the following PostgreSQL table of records, your job is to
write a SQL query that suits the user's request.

Database schema:

{DB_SCHEMA}

today's date = {date.today()}

{format_as_xml(SQL_EXAMPLES)}
"""


@agent.output_validator
async def validate_output(ctx: RunContext[Deps], output: Response) -> Response:
    if isinstance(output, InvalidRequest):
        return output

    # gemini often adds extraneous backslashes to SQL
    output.sql_query = output.sql_query.replace('\\', '')
    if not output.sql_query.upper().startswith('SELECT'):
        raise ModelRetry('Please create a SELECT query')

    try:
        await ctx.deps.conn.execute(f'EXPLAIN {output.sql_query}')
    except asyncpg.exceptions.PostgresError as e:
        raise ModelRetry(f'Invalid query: {e}') from e
    else:
        return output


async def main():
    if len(sys.argv) == 1:
        prompt = 'show me logs from yesterday, with level "error"'
    else:
        prompt = sys.argv[1]

    async with database_connect(
        'postgresql://postgres:postgres@localhost:5432', 'postgres'
    ) as conn:
        deps = Deps(conn)
        result = await agent.run(prompt, deps=deps)

        # Execute the query if generation was successful
        if isinstance(result.output, Success):
            query = result.output.sql_query
            # Limit results to avoid excessive output
            if 'LIMIT' not in query.upper():
                query += ' LIMIT 100'
            rows = await conn.fetch(query)
            result.output.results = [dict(row) for row in rows]
            
            # If no results found, extract columns from the query and show distinct values
            if not result.output.results:
                with logfire.span('fetch_distinct_values'):
                    # First, get all available columns from the table
                    try:
                        all_columns = await conn.fetch(
                            "SELECT column_name FROM information_schema.columns WHERE table_name = 'records'"
                        )
                        all_column_names = [row['column_name'] for row in all_columns]
                    except asyncpg.exceptions.PostgresError:
                        all_column_names = []
                    
                    # Extract columns from the query's WHERE clause
                    query_lower = query.lower()
                    where_part = ""
                    if 'where' in query_lower:
                        where_part = query_lower.split('where')[1]
                        if 'limit' in where_part:
                            where_part = where_part.split('limit')[0]
                        if 'order by' in where_part:
                            where_part = where_part.split('order by')[0]
                        if 'group by' in where_part:
                            where_part = where_part.split('group by')[0]
                    
                    # Find columns mentioned in the WHERE clause
                    columns_to_check = []
                    for column in all_column_names:
                        if column in where_part:
                            columns_to_check.append(column)
                    
                    # If extracting from WHERE clause didn't work, look at attributes or jsonb access
                    if not columns_to_check and "attributes->" in where_part:
                        try:
                            # Get all unique keys in the attributes JSONB column
                            json_keys = await conn.fetch(
                                "SELECT DISTINCT jsonb_object_keys(attributes) as key FROM records WHERE attributes IS NOT NULL"
                            )
                            if json_keys:
                                columns_to_check.append('attributes')
                        except asyncpg.exceptions.PostgresError:
                            pass
                    
                    # If still no columns, add default important columns
                    if not columns_to_check:
                        default_important = ['service_name', 'level', 'tags']
                        columns_to_check = [col for col in default_important if col in all_column_names]
                    
                    # Get distinct values for identified columns
                    distinct_values = {}
                    
                    for column in columns_to_check:
                        try:
                            if column == 'attributes':
                                # For JSONB column, show the available keys
                                distinct_vals = await conn.fetch(
                                    "SELECT DISTINCT jsonb_object_keys(attributes) as key FROM records WHERE attributes IS NOT NULL LIMIT 20"
                                )
                                if distinct_vals:
                                    distinct_values['attributes.keys'] = [row['key'] for row in distinct_vals]
                            else:
                                # For regular columns, show distinct values
                                distinct_vals = await conn.fetch(f"SELECT DISTINCT {column} FROM records LIMIT 20")
                                if distinct_vals:
                                    distinct_values[column] = [dict(row)[column] for row in distinct_vals]
                        except asyncpg.exceptions.PostgresError:
                            continue
                    
                    if distinct_values:
                        result.output.explanation += "\n\n**No results found for your query. Here are available values you can search for:**\n\n"
                        for column, values in distinct_values.items():
                            if column == 'attributes.keys':
                                result.output.explanation += f"\n**Available attribute keys**: {', '.join(repr(v) for v in values if v is not None)}\n"
                            else:
                                # Special handling for array types like tags
                                if column == 'tags' and values and isinstance(values[0], list):
                                    # Flatten the list of arrays
                                    all_tags = set()
                                    for tag_list in values:
                                        if tag_list:
                                            all_tags.update(tag_list)
                                    result.output.explanation += f"\n**{column}**: {', '.join(repr(v) for v in all_tags if v is not None)}\n"
                                else:
                                    result.output.explanation += f"\n**{column}**: {', '.join(repr(v) for v in values if v is not None)}\n"

    debug(result.output)


# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
@asynccontextmanager
async def database_connect(server_dsn: str, database: str) -> AsyncGenerator[Any, None]:
    with logfire.span('check and create DB'):
        conn = await asyncpg.connect(server_dsn)
        try:
            db_exists = await conn.fetchval(
                'SELECT 1 FROM pg_database WHERE datname = $1', database
            )
            if not db_exists:
                await conn.execute(f'CREATE DATABASE {database}')
        finally:
            await conn.close()

    conn = await asyncpg.connect(f'{server_dsn}/{database}')
    try:
        with logfire.span('create schema'):
            async with conn.transaction():
                if not db_exists:
                    await conn.execute(
                        "CREATE TYPE log_level AS ENUM ('debug', 'info', 'warning', 'error', 'critical')"
                    )
                    await conn.execute(DB_SCHEMA)
        yield conn
    finally:
        await conn.close()


if __name__ == '__main__':
    asyncio.run(main())
