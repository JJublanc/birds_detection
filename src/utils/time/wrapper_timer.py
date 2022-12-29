import time


def timer_wrapper(func):
    def wrapper(
        input_params: dict = {},
        timer_key: str = "Elapsed time",
        *args,
        **kwargs,
    ):
        start_load = time.perf_counter()
        results = func(*args, **kwargs)
        end_load = time.perf_counter()
        input_params[timer_key] = end_load - start_load

        return results, input_params

    return wrapper
