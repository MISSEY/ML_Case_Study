'''
    In this dag we are going to see the usage of
    1. KubernetesPodOperator where we want to execute tasks based on our custom docker-image -- usecase
    2. PythonOperator to call python functions - Based on Airflow native docker image
    3. Skip tasks using AirflowSkipException - to skip certain tasks based on some condition: for eg: if we are training a
          classifier for the first time then the task on testing the older model should executed, so we would like skip and continue only with training.
    4. BashOperator to execute bash commands in the docker container.
'''

import glob
import os
from datetime import datetime

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.models.param import Param
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.python import PythonOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule
# from kubernetes import client, config, utils
from kubernetes.client import models as k8s

# set the default arguments for dags , please refer the airflow for more.
default_args = {
    "owner": "Saurabh Mishra",
    'start_date': days_ago(31),
    'email': ['sami02@dfki.de'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    # 'retry_delay': timedelta(minutes=2),
}


def compare_models(**context):
    '''
        compare old and new models
    '''

    task_id = context['task_id']
    task_instance = context["dag_run"].get_task_instance(task_id=task_id)

    # check whether the test_old_model is skipped or not as it might be that you are training for the first time
    if task_instance.state == 'skipped':
        return 'task_first_model'
    else:
        new_model_accu = context['ti'].xcom_pull(task_ids='task_test_new_model')
        old_model_accu = context['ti'].xcom_pull(task_ids='task_test_old_model')
        if new_model_accu > old_model_accu:
            return 'task_add_new_model'
        else:
            return 'task_retrieve_old_model'




# give dag_id according to https://git.ni.dfki.de/ml_infrastructure/airflow-pbr-dags/-/blob/main/docs/guideline.md

if k8s:
    with DAG(
            dag_id='Room_Occupancy_XgBoost',
            schedule_interval=None,
            start_date=datetime(2021, 1, 1),
            catchup=False,
            tags=['machine_learning'],
            render_template_as_native_obj=True,
            # this if you want to push a dictionary to Xcom table https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/xcoms.html#xcoms
            default_args=default_args,
    ) as dag:
        mount_dataset_model = "/home/dvc/ml_pipeline/"

        # https://kubernetes.io/docs/concepts/storage/persistent-volumes/#reserving-a-persistentvolume
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
            k8s.V1VolumeMount(mount_path=mount_dataset_model, name="create-pv-dvc-model"),
        ]

        # Environment variable for Pods and containers
        AWS_KEY_ENV = k8s.V1EnvVar(name='AWS_ACCESS_KEY_ID', value_from=k8s.V1EnvVarSource(
            secret_key_ref=k8s.V1SecretKeySelector(name='local-public-dvc-access',
                                                   key='AWS_ACCESS_KEY_ID')))
        AWS_SECRET_ENV = k8s.V1EnvVar(name='AWS_SECRET_ACCESS_KEY', value_from=k8s.V1EnvVarSource(
            secret_key_ref=k8s.V1SecretKeySelector(name='local-public-dvc-access',
                                                   key='AWS_SECRET_ACCESS_KEY')))
        
        GIT_TOKEN = k8s.V1EnvVar(name='GIT_TOKEN', value_from=k8s.V1EnvVarSource(
            secret_key_ref=k8s.V1SecretKeySelector(name='github-saurabh-token',
                                                   key='GIT_TOKEN')))
        
        GIT_USERNAME = k8s.V1EnvVar(name='GIT_USERNAME', value='MISSEY')
      
        '''
        KubernetesPodOperator for custom docker image
        '''

        task_download_dataset = KubernetesPodOperator(
            namespace='airflow',
            image="smishra03/dvc:latest",
            arguments=[
                "-g", 'github.com/MISSEY/ML_Case_Study.git',
                "-d", 'occupancy_data.dvc',
                "-v", f"{mount_dataset_model}"],
            labels={"download dataset": "dvc"},
            volume_mounts=volume_mounts,
            volumes=volumes,
            env_vars=[AWS_KEY_ENV, AWS_SECRET_ENV,GIT_TOKEN,GIT_USERNAME],
            name="Download_dataset_using_dvc",
            task_id="task_download_dataset",
            is_delete_operator_pod=True,
            get_logs=True,
            do_xcom_push=False,
            container_security_context={
                "privileged": True,
                "allow_privilege_escalation": True,
            }
        
        )

        task_download_model = KubernetesPodOperator(
            namespace='airflow',
            image="smishra03/dvc:latest",
            arguments=[
                "-g", 'github.com/MISSEY/ML_Case_Study.git',
                "-d", 'models.dvc',
                "-v", f"{mount_dataset_model}"],
            labels={"download dataset": "dvc"},
            volume_mounts=volume_mounts,
            volumes=volumes,
            env_vars=[AWS_KEY_ENV, AWS_SECRET_ENV,GIT_TOKEN,GIT_USERNAME],
            name="Download_dataset_using_dvc",
            task_id="task_download_dataset",
            is_delete_operator_pod=True,
            get_logs=True,
            do_xcom_push=False,
            container_security_context={
                "privileged": True,
                "allow_privilege_escalation": True,
            }
        
        )
    
       
        task_test_old_model = KubernetesPodOperator(
            namespace='airflow',
            image="smishra03/occupancy-training",
            arguments=[
                "-e",
                "-m", f'{mount_dataset_model}/ML_Case_Study/models',
                "-d", f'{mount_dataset_model}/ML_Case_Study/occupancy_data'
                ],
            labels={"test_model": "xgboost"},
            volume_mounts=volume_mounts,
            volumes=volumes,
            env_vars=[AWS_KEY_ENV, AWS_SECRET_ENV],
            name="test_the_old_model",
            task_id="test_old_model",
            is_delete_operator_pod=True,
            get_logs=True,
            do_xcom_push=True,
            container_security_context={
                "privileged": True,  # same as docker run --privileged
                "allow_privilege_escalation": True,
            }

        )

        task_train_new_model = KubernetesPodOperator(
            namespace='airflow',
            image="smishra03/occupancy-training",
            arguments=[
                "-m", f'{mount_dataset_model}/ML_Case_Study/models',
                "-d", f'{mount_dataset_model}/ML_Case_Study/occupancy_data'
                ],
            labels={"train_model": "xgboost"},
            volume_mounts=volume_mounts,
            volumes=volumes,
            env_vars=[AWS_KEY_ENV, AWS_SECRET_ENV],
            name="task_train_new_model",
            task_id="train_model",
            is_delete_operator_pod=True,
            get_logs=True,
            do_xcom_push=False,
            container_security_context={
                "privileged": True,
                "allow_privilege_escalation": True,
            },
            trigger_rule=TriggerRule.NONE_FAILED
            # trigger anyway for training if  previous task is success or skipped (None failed)
        )

        task_test_new_model = KubernetesPodOperator(
            namespace='airflow',
            image="smishra03/occupancy-training",
            arguments=[
                "-e",
                "-m", f'{mount_dataset_model}/ML_Case_Study/models',
                "-d", f'{mount_dataset_model}/ML_Case_Study/occupancy_data'
                ],
            labels={"test_model": "xgboost"},
            volume_mounts=volume_mounts,
            volumes=volumes,
            env_vars=[AWS_KEY_ENV, AWS_SECRET_ENV],
            name="task_test_new_model",
            task_id="test_new_trained_model",
            is_delete_operator_pod=True,
            get_logs=True,
            do_xcom_push=False,
            container_security_context={
                "privileged": True,
                "allow_privilege_escalation": True,
            },
            trigger_rule=TriggerRule.NONE_FAILED

        )

        task_compare_previous = BranchPythonOperator(
            task_id='compare_both_versions_old_and_new',
            python_callable=compare_models,
            provide_context=True,
            trigger_rule=TriggerRule.NONE_FAILED,
            op_kwargs={'task_id': 'task_test_old_model'}  # task id to check if task_test_old_model is skipped

        )

        task_add_new_model = BashOperator(
            task_id='select_new_model_version',
            bash_command="echo newer model is better than older",
            # cwd=os.path.abspath(mount_model),
            trigger_rule=TriggerRule.NONE_FAILED,
            # executor_config=executor_config_for_mount
            # executor config which overrides the and create volume mounts inside container.

        )

        task_retrieve_old_model = BashOperator(
            task_id='retrieve_old_model_version',
            bash_command="echo older model is better than newer",
            # cwd=os.path.abspath(mount_model),
            trigger_rule=TriggerRule.NONE_FAILED,
            # executor_config=executor_config_for_mount
            # executor config which overrides the and create volume mounts inside container.
        )

        task_first_model = BashOperator(
            task_id='The_first_trained_model',
            bash_command="echo first model is ready to deploy",
            # cwd=os.path.abspath(mount_model),
            trigger_rule=TriggerRule.NONE_FAILED,
            # executor_config=executor_config_for_mount
            # executor config which overrides the and create volume mounts inside container.
        )

        task_download_dataset >> task_download_model >>  task_test_old_model >> task_train_new_model >> task_test_new_model >> task_compare_previous >> [
            task_add_new_model, task_retrieve_old_model, task_first_model]