from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# A simple Python function
def say_hello():
    print("Hello Airflow!")

# DAG definition
with DAG(
    dag_id='hello_airflow',
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False
) as dag:

    hello_task = PythonOperator(
        task_id='say_hello_task',
        python_callable=say_hello
    )
