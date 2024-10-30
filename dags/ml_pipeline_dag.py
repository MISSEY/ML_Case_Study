import os
from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule
from docker.types import Mount

default_args = {
    "owner": "Saurabh Mishra",
    'start_date': days_ago(31),
    'email': ['saurabhmisra0109@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    # 'retry_delay': timedelta(minutes=2),
}



with DAG(
        dag_id='ml_pipeline_dag',
        schedule_interval=None,
        start_date=datetime(2024, 10, 31),
        catchup=False,
        tags=['machine_learning'],
        render_template_as_native_obj=True,
        # this if you want to push a dictionary to Xcom table https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/xcoms.html#xcoms
        default_args=default_args,
) as dag:
        
        training = DockerOperator(
            image="smishra03/occupancy-training:latest",
            command =["cd ml_pipeline; python occupancy.py"],
            # labels={"ml_training": "xgboost"},
            # name="Training",
            task_id="XgBoost_model_training",
            container_name='occupancy-training',
            network_mode='host',
            xcom_all=True,
            api_version='auto',
            auto_remove=True,
            mount_tmp_dir=False,
            mounts=[
                    Mount(target='/ml_pipeline/',source='/mnt/d/personal_projects/ML_Case_Study')
            ],
            tty=True,
            # docker_url='unix:///var/run/docker.sock',
            privileged=True
        )

        # @task.docker(image="python:3.9-slim-bookworm", multiple_outputs=True)
        # def transform(order_data_dict: dict):
        #     """
        #     #### Transform task
        #     A simple Transform task which takes in the collection of order data and
        #     computes the total order value.
        #     """
        #     total_order_value = 0

        #     for value in order_data_dict.values():
        #         total_order_value += value

        #     return {"total_order_value": total_order_value}

        training
