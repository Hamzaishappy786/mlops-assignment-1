from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import json

def load_data(**kwargs):
    print("Loading dataset...")
    data = {"feature1": [10, 20, 30], "target": [0, 1, 0]}
    print(f"Data loaded successfully: {data}")
    return json.dumps(data)

def train_model(ti):
    data_str = ti.xcom_pull(task_ids='load_data_task')
    data = json.loads(data_str)

    print("Training model...")
    print(f"Training on features: {data['feature1']}")

    accuracy = 0.92
    print(f"Model training complete. Accuracy: {accuracy}")
    return accuracy


def save_model(ti):
    accuracy = ti.xcom_pull(task_ids='train_model_task')

    print(f"Saving model with accuracy: {accuracy}...")
    print("Model saved to model_registry/v1/")

default_args = {
    'owner': 'mlops_student',
    'start_date': datetime(2023, 1, 1),
    'retries': 0
}

with DAG('train_pipeline',
         default_args=default_args,
         schedule_interval=None,
         catchup=False) as dag:

    load_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data
    )

    train_task = PythonOperator(
        task_id='train_model_task',
        python_callable=train_model
    )

    save_task = PythonOperator(
        task_id='save_model_task',
        python_callable=save_model
    )

    load_task >> train_task >> save_task