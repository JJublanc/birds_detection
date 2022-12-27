# mlops_template

Source : https://www.tensorflow.org/hub/tutorials/tf2_object_detection

## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.com/littlebigcode/templates/mlops_template.git
git branch -M main
git push -uf origin main
```

# Setting
Make you have installed :
python > 3.8
poetry > 1.2.2

## Set up your environments - Automatic
Setting up for the first time
```
make hello-world
```
=> This will clean your project, install python with pyenv and install a dev and prod env.

>⚠️ You may need to source your terminal and update your IDE interpreter with the dev env. ⚠️
# Set up your environments - Manual
Create the new virtual env
```
python -m venv venv
```
And activate it. On Windows it will be
```
source venv/Scripts/activate
```
If necessary append you PYTHONPATH
```
export PYTHONPATH="${PYTHONPATH}:."
```

With conda

```
conda create -p ./venv_dev python=3.7 --no-default-packages
conda activate ./venv_dev
pip install -r requirements/dev.txt
conda-develop .
```

# Set your Pycharm interpreter

## Create a file .env at the root of your project
First run `az login`

Add the following values
```
API_HOST=  # for instance "127.0.0.1"
API_PORT=  # for instance 8000
RESOURCE_GROUP=
SUBSCRIPTION= # for instance "Azure Sandbox"
SUBSCRIPTION_ID= #
WORKSPACE_NAME= #
WORKSPACE_CONFIG_FILENAME= # for instance "workspace_config"
SERVICE_PRINCIPAL= # for instance "service_principal"
EXPERIMENT_NAME=
```

## Install your pre-commit
```
pre-commit install
pre-commit install --hook-type commit-msg
```

Now you can run pre-commit on all your files to check your code
```
pre-commit run --all-files
```

And each time you commit your code, pre-commit checks will be made.

## Create your Azure resource to track your training

```
make show_subscriptions
make set_subscription
make create_resourcegroup
make create_service_principal
make create_workspace_azure_ml
```

## Train your model
```
python src/train/task.py
```
It will save the model in a folder named `model` and track the result using mlflow and your Azure ML Workspace.
You can go there to check your experiments.

### Deploy your app

__Locally__
```
python app/main.py
```

__Using Docker__
Make sure make is installed on your machine : https://www.gnu.org/software/make/
```
make build
make run_app
```

__Test API__
Then test your API
```
http://127.0.0.1:8000/docs
```

# Download data from Kaggle

Ensure your key kaggle.json is in the location ~/.kaggle/kaggle.json to use the API. To get your key,
go to your profile on kaggle.

```
pip install kaggle
kaggle datasets download -d sumanthvrao/daily-climate-time-series-data
unzip daily-climate-time-series-data.zip -d data/
rm -f daily-climate-time-series-data.zip
```

# Fix pre-commit issues
isort --profile black -l 79 -v --skip env_test --skip venv
black -l 79 src
flake8 --ignore W191 --extend-ignore E203 --exclude venv --exclude env-test src

# TODO
- Fix env names test should be dev
- Add env logic test in dev first then deploy on prod if valid
