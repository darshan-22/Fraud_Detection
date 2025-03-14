import sqlite3
import pandas as pd

# Connect to your SQLite database
conn = sqlite3.connect('test.db')

# List all tables in the database
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in the database:", tables)

# Choose a table to view
table_name = input("Enter the table name to view data: ")

# Load table data into a DataFrame
df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

# Display the first 10 rows
print(f"Data from {table_name} table:")
print(df.head(10))

# Close the database connection
conn.close()