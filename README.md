# How to deploy using a managed online endpoint

This project shows how to train a Fashion MNIST model locally, and how to deploy it using an online managed endpoint, without using MLflow.

## Blog post

To learn more about the code in this repo, check out the accompanying blog post: https://bea.stollnitz.com/blog/aml-online-endpoint-no-mlflow/

## Setup

- You need to have an Azure subscription. You can get a [free subscription](https://azure.microsoft.com/en-us/free) to try it out.
- Create a [resource group](https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal).
- Create a new machine learning workspace by following the "Create the workspace" section of the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources). Keep in mind that you'll be creating a "machine learning workspace" Azure resource, not a "workspace" Azure resource, which is entirely different!
- Install the Azure CLI by following the instructions in the [documentation](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli).
- Install the ML extension to the Azure CLI by following the "Installation" section of the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli).
- Install and activate the conda environment by executing the following commands:

```
conda env create -f environment.yml
conda activate aml_online_endpoint_no_mlflow
```

- Within VS Code, go to the Command Palette clicking "Ctrl + Shift + P," type "Python: Select Interpreter," and select the environment that matches the name of this project.
- In a terminal window, log in to Azure by executing `az login --use-device-code`.
- Set your default subscription by executing `az account set -s "<YOUR_SUBSCRIPTION_NAME_OR_ID>"`. You can verify your default subscription by executing `az account show`, or by looking at `~/.azure/azureProfile.json`.
- Set your default resource group and workspace by executing `az configure --defaults group="<YOUR_RESOURCE_GROUP>" workspace="<YOUR_WORKSPACE>"`. You can verify your defaults by executing `az configure --list-defaults` or by looking at `~/.azure/config`.
- You can now open the [Azure Machine Learning studio](https://ml.azure.com/), where you'll be able to see and manage all the machine learning resources we'll be creating.
- Install the [Azure Machine Learning extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai), and log in to it by clicking on "Azure" in the left-hand menu, and then clicking on "Sign in to Azure."

## Training on your development machine

Under "Run and Debug" on VS Code's left navigation, choose the "Train locally" run configuration and press F5. A `model` folder is created with the trained model.

## Inference on your development machine

If you're not running this project on GitHub Codespaces, you can use a local endpoint to test deployment on your development machine:

```
cd aml_online_endpoint_no_mlflow
```

You can deploy locally to test your endpoint before you deploy in the cloud. First start Docker. Then create the local endpoint and deployment.

```
az ml online-endpoint create -f cloud/endpoint.yml --local
az ml online-deployment create -f cloud/deployment-local.yml --local
```

Verify your endpoint was created:

```
az ml online-endpoint list --local
```

Invoke the endpoint:

```
az ml online-endpoint invoke --name endpoint-online-no-mlflow --request-file test_data/images_azureml.json --local
```

Once you're done, you can delete the endpoint:

```
az ml online-endpoint delete -n endpoint-online-no-mlflow -y --local
```

If you're running this project on GitHub Codespaces, you can select the "Test locally" run configuration and press F5 to test your model on your development machine.

## Deploying in the cloud using Azure ML

```
cd aml_online_endpoint_no_mlflow
```

Create the model resource on Azure ML.

```
az ml model create --path model --name model-online-no-mlflow --version 1
```

Create the endpoint.

```
az ml online-endpoint create -f cloud/endpoint.yml
az ml online-deployment create -f cloud/deployment.yml --all-traffic
```

Invoke the endpoint.

```
az ml online-endpoint invoke --name endpoint-online-no-mlflow --request-file test_data/images_azureml.json
```

Clean up the endpoint, to avoid getting charged.

```
az ml online-endpoint delete --name endpoint-online-no-mlflow -y
```

## Related resources

- [Local endpoints](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-managed-online-endpoints#deploy-and-debug-locally-by-using-local-endpoints?WT.mc_id=aiml-69969-bstollnitz)
- [Deploy using managed online endpoints](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-managed-online-endpoints?WT.mc_id=aiml-69969-bstollnitz)
