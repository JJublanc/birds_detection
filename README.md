# mlops_template

Source : https://www.tensorflow.org/hub/tutorials/tf2_object_detection

## Getting started
requirement : poetry, pyenv
```
pyenv install 3.9.1
pyenv local 3.9.1
poetry config virtualenvs.in-project true
pyenv which python | xargs poetry env use
poetry install
```

# Add additional package
requirement : protobut-compiler
```
git clone --depth 1 https://github.com/tensorflow/models
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cd ../..
poetry add ./models/research/
```
# Known issue
A file is missing in protobuf package : just add it as explain here :
https://stackoverflow.com/questions/71759248/importerror-cannot-import-name-builder-from-google-protobuf-internal

# Process images
```
poetry run python src/predict/example_object_detection.py
```

# Launch app
```
poetry run python -m streamlit app/dashboard.py
```
