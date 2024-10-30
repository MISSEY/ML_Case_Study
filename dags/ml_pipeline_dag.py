"""
Room Occupancy Classification ML Pipeline DAG

This DAG implements a complete machine learning pipeline for room occupancy classification using XGBoost.
It handles dataset and model management using DVC, model training, testing, and version control.

Key Features:
- Downloads datasets and models using DVC
- Tests existing model performance (if available)
- Trains new model
- Compares model performances
- Manages model versioning

Components:
1. KubernetesPodOperator: Executes tasks in custom Docker containers
2. PythonOperator: Executes Python functions
3. BashOperator: Executes bash commands
4. Conditional task skipping using AirflowSkipException

Prerequisites:
- Kubernetes cluster with Airflow installed
- DVC setup with Minio S3 backend
- Custom Docker images:
  - smishra03/dvc:latest
  - smishra03/occupancy-training
- Required Kubernetes secrets:
  - local-public-dvc-access
  - github-saurabh-token
"""

import os
from datetime import datetime

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule
from kubernetes.client import models as k8s

# DAG default configuration
default_args = {
    "owner": "Saurabh Mishra",
    'start_date': days_ago(31),
    'email': ['sami02@dfki.de'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
}

def compare_models(**context) -> str:
    """
    Compares the performance of old and new models based on ROC-AUC scores.
    
    This function is called after both models have been tested. It checks if:
    1. This is the first model being trained (no old model exists)
    2. If both models exist, compares their ROC-AUC scores
    
    Args:
        **context: Airflow context containing task instance and DAG run information
        
    Returns:
        str: Task ID of the next task to execute based on comparison:
            - 'The_first_trained_model': If no old model exists
            - 'select_new_model_version': If new model performs better
            - 'retrieve_old_model_version': If old model performs better
    """
    dag_run = context["dag_run"]
    task_instance = dag_run.get_task_instance(task_id='test_old_model')

    # Check if this is the first model (no old model to compare against)
    if task_instance.state == 'skipped':
        return 'The_first_trained_model'
    
    # Compare model performances
    new_model_accu = context['ti'].xcom_pull(task_ids='test_new_trained_model')['roc_auc']
    old_model_accu = context['ti'].xcom_pull(task_ids='test_old_model')['roc_auc']
    
    return 'select_new_model_version' if new_model_accu > old_model_accu else 'retrieve_old_model_version'

# Initialize DAG
with DAG(
        dag_id='Room_Occupancy_XgBoost',
        schedule_interval=None,
        start_date=datetime(2021, 1, 1),
        catchup=False,
        tags=['machine_learning'],
        render_template_as_native_obj=True,
        default_args=default_args,
) as dag:
    # Define common paths and configurations
    MOUNT_PATH = "/home/dvc/ml_pipeline/"
    
    # Configure Kubernetes volumes and mounts
    volumes = [
        k8s.V1Volume(
            name='create-pv-dvc-model',
            persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                claim_name='pvc-dvc-model',
                read_only=False
            )
        ),
    ]

    volume_mounts = [
        k8s.V1VolumeMount(mount_path=MOUNT_PATH, name="create-pv-dvc-model"),
    ]

    # Define environment variables for AWS and Git authentication
    env_vars = [
        k8s.V1EnvVar(
            name='AWS_ACCESS_KEY_ID',
            value_from=k8s.V1EnvVarSource(
                secret_key_ref=k8s.V1SecretKeySelector(
                    name='local-public-dvc-access',
                    key='AWS_ACCESS_KEY_ID'
                )
            )
        ),
        k8s.V1EnvVar(
            name='AWS_SECRET_ACCESS_KEY',
            value_from=k8s.V1EnvVarSource(
                secret_key_ref=k8s.V1SecretKeySelector(
                    name='local-public-dvc-access',
                    key='AWS_SECRET_ACCESS_KEY'
                )
            )
        ),
        k8s.V1EnvVar(
            name='GIT_TOKEN',
            value_from=k8s.V1EnvVarSource(
                secret_key_ref=k8s.V1SecretKeySelector(
                    name='github-saurabh-token',
                    key='GIT_TOKEN'
                )
            )
        ),
        k8s.V1EnvVar(name='GIT_USERNAME', value='MISSEY')
    ]

    # Security context for containers
    security_context = {
        "privileged": True,
        "allow_privilege_escalation": True,
    }

    # Task 1: Download dataset using DVC
    task_download_dataset = KubernetesPodOperator(
        namespace='airflow',
        image="smishra03/dvc:latest",
        arguments=[
            "-g", 'github.com/MISSEY/ML_Case_Study.git',
            "-d", 'occupancy_data.dvc',
            "-v", f"{MOUNT_PATH}"
        ],
        labels={"download_dataset": "dvc"},
        volume_mounts=volume_mounts,
        volumes=volumes,
        env_vars=env_vars,
        name="Download_dataset_using_dvc",
        task_id="task_download_dataset",
        is_delete_operator_pod=True,
        get_logs=True,
        do_xcom_push=False,
        container_security_context=security_context,
        image_pull_policy='Always'
    )

    # Task 2: Download existing model (if any) using DVC
    task_download_model = KubernetesPodOperator(
        namespace='airflow',
        image="smishra03/dvc:latest",
        arguments=[
            "-g", 'github.com/MISSEY/ML_Case_Study.git',
            "-d", 'models.dvc',
            "-v", f"{MOUNT_PATH}"
        ],
        labels={"download_model": "dvc"},
        volume_mounts=volume_mounts,
        volumes=volumes,
        env_vars=env_vars,
        name="Download_model_using_dvc",
        task_id="task_download_model",
        is_delete_operator_pod=True,
        get_logs=True,
        do_xcom_push=False,
        container_security_context=security_context
    )

    # Task 3: Test existing model performance
    task_test_old_model = KubernetesPodOperator(
        namespace='airflow',
        image="smishra03/occupancy-training",
        arguments=[
            "-e",
            "-m", f'{MOUNT_PATH}/ML_Case_Study/models',
            "-d", f'{MOUNT_PATH}/ML_Case_Study/occupancy_data'
        ],
        labels={"test_model": "xgboost"},
        volume_mounts=volume_mounts,
        volumes=volumes,
        env_vars=env_vars[:2],  # Only Minio credentials needed
        name="test_old_model",
        task_id="test_old_model",
        is_delete_operator_pod=True,
        get_logs=True,
        do_xcom_push=True,
        container_security_context=security_context,
        image_pull_policy='Always'
    )

    # Task 4: Train new model
    task_train_new_model = KubernetesPodOperator(
        namespace='airflow',
        image="smishra03/occupancy-training",
        arguments=[
            "-m", f'{MOUNT_PATH}/ML_Case_Study/models',
            "-d", f'{MOUNT_PATH}/ML_Case_Study/occupancy_data'
        ],
        labels={"train_model": "xgboost"},
        volume_mounts=volume_mounts,
        volumes=volumes,
        env_vars=env_vars[:2],
        name="train_new_model",
        task_id="train_model",
        is_delete_operator_pod=True,
        get_logs=True,
        do_xcom_push=False,
        container_security_context=security_context,
        trigger_rule=TriggerRule.NONE_FAILED  # Continue if previous task succeeded or was skipped
    )

    # Task 5: Test newly trained model
    task_test_new_model = KubernetesPodOperator(
        namespace='airflow',
        image="smishra03/occupancy-training",
        arguments=[
            "-e",
            "-m", f'{MOUNT_PATH}/ML_Case_Study/models',
            "-d", f'{MOUNT_PATH}/ML_Case_Study/occupancy_data'
        ],
        labels={"test_model": "xgboost"},
        volume_mounts=volume_mounts,
        volumes=volumes,
        env_vars=env_vars[:2],
        name="test_new_model",
        task_id="test_new_trained_model",
        is_delete_operator_pod=True,
        get_logs=True,
        do_xcom_push=True,
        container_security_context=security_context,
        trigger_rule=TriggerRule.NONE_FAILED
    )

    # Task 6: Compare model performances
    task_compare_previous = BranchPythonOperator(
        task_id='compare_both_versions_old_and_new',
        python_callable=compare_models,
        provide_context=True,
        trigger_rule=TriggerRule.NONE_FAILED,
        op_kwargs={'task_id': 'task_test_old_model'}
    )

    # Final tasks based on comparison results
    task_add_new_model = BashOperator(
        task_id='select_new_model_version',
        bash_command="echo 'New model performance is superior - proceeding with deployment'",
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    task_retrieve_old_model = BashOperator(
        task_id='retrieve_old_model_version',
        bash_command="echo 'Existing model performs better - keeping current deployment'",
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    task_first_model = BashOperator(
        task_id='The_first_trained_model',
        bash_command="echo 'First model trained successfully - ready for deployment'",
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    # Define task dependencies
    task_download_dataset >> task_download_model >> task_test_old_model >> \
    task_train_new_model >> task_test_new_model >> task_compare_previous >> \
    [task_add_new_model, task_retrieve_old_model, task_first_model]