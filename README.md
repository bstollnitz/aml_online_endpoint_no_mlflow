# How to deploy using a managed online endpoint

This project shows how to train a Fashion MNIST model locally, and how to deploy it using an online managed endpoint, without using MLflow. 


## Blog post

To learn more about the code in this repo, check out the accompanying blog post: https://bea.stollnitz.com/blog/aml-online-endpoint-no-mlflow/


## Azure setup

* You need to have an Azure subscription. You can get a [free subscription](https://azure.microsoft.com/en-us/free?WT.mc_id=aiml-69969-bstollnitz) to try it out.
* Create a [resource group](https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal?WT.mc_id=aiml-69969-bstollnitz).
* Create a new machine learning workspace by following the "Create the workspace" section of the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources?WT.mc_id=aiml-69969-bstollnitz). Keep in mind that you'll be creating a "machine learning workspace" Azure resource, not a "workspace" Azure resource, which is entirely different!
* If you have access to GitHub Codespaces, click on the "Code" button in this GitHub repo, select the "Codespaces" tab, and then click on "New codespace."
* Alternatively, if you plan to use your local machine:
  * Install the Azure CLI by following the instructions in the [documentation](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?WT.mc_id=aiml-69969-bstollnitz).
  * Install the ML extension to the Azure CLI by following the "Installation" section of the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?WT.mc_id=aiml-69969-bstollnitz).
* In a terminal window, login to Azure by executing `az login --use-device-code`. 
* Set your default subscription by executing `az account set -s "<YOUR_SUBSCRIPTION_NAME_OR_ID>"`. You can verify your default subscription by executing `az account show`, or by looking at `~/.azure/azureProfile.json`.
* Set your default resource group and workspace by executing `az configure --defaults group="<YOUR_RESOURCE_GROUP>" workspace="<YOUR_WORKSPACE>"`. You can verify your defaults by executing `az configure --list-defaults` or by looking at `~/.azure/config`.
* You can now open the [Azure Machine Learning studio](https://ml.azure.com/?WT.mc_id=aiml-69969-bstollnitz), where you'll be able to see and manage all the machine learning resources we'll be creating.
* Although not essential to run the code in this post, I highly recommend installing the [Azure Machine Learning extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai).



## Project setup

If you have access to GitHub Codespaces, click on the "Code" button in this GitHub repo, select the "Codespaces" tab, and then click on "New codespace."

Alternatively, you can set up your local machine using the following steps.

Install conda environment:

```
conda env create -f environment.yml
```

Activate conda environment:

```
conda activate aml_online_endpoint_no_mlflow
```


## Training on your development machine

Open the `src/train.py` file and press F5. A `model` folder is created with the trained model.


## Inference on your development machine

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

* [Local endpoints](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-managed-online-endpoints#deploy-and-debug-locally-by-using-local-endpoints?WT.mc_id=aiml-69969-bstollnitz)
* [Deploy using managed online endpoints](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-managed-online-endpoints?WT.mc_id=aiml-69969-bstollnitz)
