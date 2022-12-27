import logging

logging.basicConfig(level=logging.INFO)


def get_prediction_model():
    def get_foo_model():
        return 0

    return get_foo_model


def predict(model_predictor):
    return model_predictor()


if __name__ == "__main__":
    # Get the best model
    model = get_prediction_model()

    # Make a prediction
    logging.info(f"Model prediction = {predict(model)}")
