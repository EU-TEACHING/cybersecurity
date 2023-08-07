import os
import mlflow
from mlflow.tracking import MlflowClient
import tensorflow as tf
from dotenv import load_dotenv


def connect_to_mlflow(mlflow_cfg):
    # Load environment variables from .env file
    load_dotenv()

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow_s3_endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    if mlflow_s3_endpoint_url and aws_access_key_id and aws_secret_access_key:
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = mlflow_s3_endpoint_url
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

    experiment_name = mlflow_cfg["experiment_name"]
    experiment_id = retrieve_mlflow_experiment_id(experiment_name, create=True)

    return mlflow.start_run(experiment_id=experiment_id)


def retrieve_mlflow_experiment_id(name, create=False):
    experiment_id = None
    if name:
        existing_experiment = MlflowClient().get_experiment_by_name(name)
        if existing_experiment and existing_experiment.lifecycle_stage == 'active':
            experiment_id = existing_experiment.experiment_id
        else:
            if create:
                experiment_id = mlflow.create_experiment(name)
            else:
                raise Exception(f'Experiment "{name}" not found in {mlflow.get_tracking_uri()}')

    if experiment_id is not None:
        experiment = MlflowClient().get_experiment(experiment_id)
        print("Experiment name: {}".format(experiment.name))
        print("Experiment_id: {}".format(experiment.experiment_id))
        print("Artifact Location: {}".format(experiment.artifact_location))
        print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

    return experiment_id


def log_mlflow_metrics(storage, accuracy, precision, recall, f1, conf_matrix, mode):
    mlflow.log_metric(f'Accuracy_{mode}', accuracy)
    mlflow.log_metric(f'Precision_{mode}', precision)
    mlflow.log_metric(f'Recall_{mode}', recall)
    mlflow.log_metric(f'F1_{mode}', f1)

    write_confusion_matrix_to_md(conf_matrix, f'{storage}/confusion_matrix_{mode}.md')
    # Log the entire confusion matrix as a single artifact
    mlflow.log_artifact(f'{storage}/confusion_matrix_{mode}.md')

    # # Flatten the confusion matrix
    # flattened_conf_matrix = conf_matrix.flatten()
    #
    # # Log the individual elements of the confusion matrix as metrics
    # mlflow.log_metric(f"TN_{mode}", flattened_conf_matrix[0])
    # mlflow.log_metric(f"FP_{mode}", flattened_conf_matrix[1])
    # mlflow.log_metric(f"FN_{mode}", flattened_conf_matrix[2])
    # mlflow.log_metric(f"TP_{mode}", flattened_conf_matrix[3])


def write_confusion_matrix_to_md(conf_matrix, md_file):
    # Open the .md file in write mode
    with open(md_file, 'w') as file:
        if conf_matrix.shape == (1, 1):
            # Handle the case of a single class (TP or TN)
            file.write(f"|       | {conf_matrix[0, 0]} |\n")
            file.write(f"|-------|-------|\n")
            file.write(f"| True 0| {conf_matrix[0, 0]} |\n")
        else:
            # Write the Markdown table header
            file.write("|       | Predicted 0 | Predicted 1 |\n")
            file.write("|-------|-------------|-------------|\n")

            # Write the table rows with confusion matrix values
            for i in range(conf_matrix.shape[0]):
                file.write(f"| True {i} | {conf_matrix[i, 0]}        | {conf_matrix[i, 1]}        |\n")


def check_gpu_usage():
    if tf.test.gpu_device_name():
        print('GPU is available and TensorFlow is using GPU.')
        return 'GPU'
    else:
        print('GPU is not available. TensorFlow is using CPU.')
        return 'CPU'
