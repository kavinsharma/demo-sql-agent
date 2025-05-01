# PydanticAI SQL Generation Demo

This project demonstrates how to use `pydantic-ai` to create an agent that can:

1.  Understand natural language requests related to querying a database.
2.  Generate SQL queries based on a provided schema.
3.  Validate the generated SQL.
4.  Execute the SQL and return the results.

## Setup

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone https://github.com/definite-app/demo-sql-agent.git
    cd demo-sql-agent
    ```

2.  **Start PostgreSQL using Docker:**
    Make sure you have Docker installed and running.
    ```bash
    # Create a directory for persistent data (optional but recommended)
    mkdir postgres-data

    # Run postgres container
    docker run --rm --name pydantic-sql-demo-db -e POSTGRES_PASSWORD=postgres -p 54320:5432 -v $(pwd)/postgres-data:/var/lib/postgresql/data postgres
    ```
    *   This uses the default password `postgres` and maps port `54320` on your host to the container's port `5432`.
    *   The `--rm` flag ensures the container is removed when stopped. Remove it if you want the container to persist.
    *   The `-v` flag mounts the `postgres-data` directory for data persistence.

3.  **Set up the Python Environment using `uv`:**
    If you don't have `uv` installed, follow the instructions at [astral.sh/uv](https://docs.astral.sh/uv/install.sh).
    ```bash
    # Create a virtual environment
    uv venv .venv

    # Install dependencies
    uv pip install -r requirements.txt
    ```
    *(Note: We haven't created requirements.txt yet, but will add it)*

4.  **Create a `.env` file:**
    Copy the example or create a new file named `.env` in the project root and add your Gemini API key:
    ```dotenv
    GEMINI_API_KEY='YOUR_GEMINI_API_KEY'
    ```
    *(This file is ignored by Git)*

## Usage

1.  **Load Sample Data (Optional but Recommended):**
    Run the data loading script. This will create the necessary table and enum type if they don't exist and insert sample records.
    ```bash
    uv run python load_data.py
    ```

2.  **Run the SQL Generation Agent:**
    Execute the main script, optionally providing a prompt as a command-line argument.
    ```bash
    # Use the default prompt ("show me logs from yesterday, with level 'error'")
    uv run python sql_gen.py

    # Provide a custom prompt
    uv run python sql_gen.py "show me records from service_name 'api-service' with the tag 'foobar'"
    ```
    The script will output the generated SQL query, an explanation, and the results fetched from the database. 