import asyncio
import asyncpg
import json
from datetime import datetime, timedelta, timezone

# Database connection details (should match sql_gen.py)
DATABASE_URL = 'postgresql://postgres:postgres@localhost:54320/pydantic_ai_sql_gen'

# Sample data
SAMPLE_RECORDS = [
    {
        'created_at': datetime.now(timezone.utc) - timedelta(days=1, hours=2),
        'start_timestamp': datetime.now(timezone.utc) - timedelta(days=1, hours=2, minutes=5),
        'end_timestamp': datetime.now(timezone.utc) - timedelta(days=1, hours=2),
        'trace_id': 'trace_1',
        'span_id': 'span_1a',
        'parent_span_id': None,
        'level': 'error',
        'span_name': 'request_handler',
        'message': 'Failed to process request',
        'attributes_json_schema': None,
        'attributes': json.dumps({'http.status_code': 500, 'user_id': 'user123'}),
        'tags': ['backend', 'critical', 'foobar'],
        'is_exception': True,
        'otel_status_message': 'Internal Server Error',
        'service_name': 'api-service'
    },
    {
        'created_at': datetime.now(timezone.utc) - timedelta(days=1, hours=1),
        'start_timestamp': datetime.now(timezone.utc) - timedelta(days=1, hours=1, minutes=2),
        'end_timestamp': datetime.now(timezone.utc) - timedelta(days=1, hours=1),
        'trace_id': 'trace_2',
        'span_id': 'span_2a',
        'parent_span_id': None,
        'level': 'info',
        'span_name': 'database_query',
        'message': 'User login successful',
        'attributes_json_schema': None,
        'attributes': json.dumps({'db.statement': 'SELECT * FROM users WHERE id = ?', 'user_id': 'user456'}),
        'tags': ['database', 'auth'],
        'is_exception': False,
        'otel_status_message': 'OK',
        'service_name': 'auth-service'
    },
    {
        'created_at': datetime.now(timezone.utc) - timedelta(minutes=30),
        'start_timestamp': datetime.now(timezone.utc) - timedelta(minutes=30, seconds=10),
        'end_timestamp': datetime.now(timezone.utc) - timedelta(minutes=30),
        'trace_id': 'trace_3',
        'span_id': 'span_3a',
        'parent_span_id': 'span_2a', # Example of parent span
        'level': 'warning',
        'span_name': 'cache_lookup',
        'message': 'Cache miss for key user:profile:user789',
        'attributes_json_schema': None,
        'attributes': json.dumps({'cache.key': 'user:profile:user789', 'user_id': 'user789'}),
        'tags': ['cache', 'performance'],
        'is_exception': False,
        'otel_status_message': 'OK',
        'service_name': 'cache-service'
    },
        {
        'created_at': datetime.now(timezone.utc) - timedelta(days=2, hours=5), # Older record
        'start_timestamp': datetime.now(timezone.utc) - timedelta(days=2, hours=5, minutes=1),
        'end_timestamp': datetime.now(timezone.utc) - timedelta(days=2, hours=5),
        'trace_id': 'trace_4',
        'span_id': 'span_4a',
        'parent_span_id': None,
        'level': 'error',
        'span_name': 'payment_processor',
        'message': 'Payment failed: Insufficient funds',
        'attributes_json_schema': None,
        'attributes': json.dumps({'error.type': 'insufficient_funds', 'user_id': 'user101', 'amount': 50.0}),
        'tags': ['billing', 'critical', 'payment'],
        'is_exception': True,
        'otel_status_message': 'Failed',
        'service_name': 'billing-service'
    },

]

async def load_sample_data():
    """Connects to the database and inserts sample records."""
    conn = None
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        print(f"Connected to database {DATABASE_URL}")

        # Prepare the INSERT statement dynamically based on keys in the first record
        # Assumes all records have the same keys
        columns = SAMPLE_RECORDS[0].keys()
        placeholders = ', '.join(f'${i+1}' for i in range(len(columns)))
        sql = f"INSERT INTO records ({', '.join(columns)}) VALUES ({placeholders})"

        # Prepare data tuples in the correct order
        data_to_insert = []
        for record in SAMPLE_RECORDS:
            data_to_insert.append(tuple(record[col] for col in columns))

        await conn.executemany(sql, data_to_insert)
        print(f"Successfully inserted {len(SAMPLE_RECORDS)} sample records.")

    except Exception as e:
        print(f"Error loading data: {e}")
    finally:
        if conn:
            await conn.close()
            print("Database connection closed.")

if __name__ == '__main__':
    print("Starting data loading script...")
    asyncio.run(load_sample_data())
    print("Data loading script finished.") 