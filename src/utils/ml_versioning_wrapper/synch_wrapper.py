import logging
import os
from typing import Any

import git
import mlflow

from src.utils.ml_versioning_wrapper.py_commit import check_branch, commit_code


def tracking_wrapper(func):
    def wrapper(
        experiment_branch: str = "main",
        wrapper_experiment_name: str = "default_model",
        tracking_uri: Any = None,
        *args,
        **kwargs,
    ):
        """
        Commit and push code when training a new model. The wrapper check that
        you are on the branch dedicated to experiments. Commit information is
        tracked with mlflow so that you can easily make the link between
        experiments and code.
        :param experiment_branch: name of the branch dedicated to experiments.
        If it does not exists, just checkout to a new branch.
        :param wrapper_experiment_name: name of the current experiment. You
        can, for instance, change this when you make major changes.
        :param tracking_uri: where to save experiments (local folder or distant
         uri)
        :param args: args of the function
        :param kwargs: kwargs of the function
        :return: wrapped training function
        """
        ##########################
        # Set repo git in python #
        ##########################
        cwd = os.getcwd()

        repo, repo_url = check_branch(experiment_branch, cwd)

        #####################
        # Set mlflow params #
        #####################
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(wrapper_experiment_name)

        with mlflow.start_run():
            ####################
            # Train your model #
            ####################

            mlflow.autolog()
            results = func(*args, **kwargs)

            # ###################
            # Log your results #
            # ####################
            run = mlflow.active_run()
            run_id = run.info.run_id

            ###############
            # Commit code #
            ###############
            # TODO handle the case when there is nothing to add
            #  (run twice the same experiment)
            try:
                commit_code(repo, f"feat(exp): run_id={run_id}")
                repo.git.push("origin", experiment_branch)

                ####################
                # Track commit url #
                ####################

                commit_url = (
                    repo_url.split("@")[1].replace(":", "/").split(".")
                )
                commit_url = ".".join(commit_url[:2]) + "/commit/"
                commit_url = commit_url + str(repo.head.commit.hexsha)
                mlflow.log_params({"commit url ": commit_url})
            except git.GitCommandError as e:
                logging.warning(e)
                logging.warning(
                    "No worries! To get back on track just solve the "
                    "GitCommand issue and start again."
                )

        return results

    return wrapper
