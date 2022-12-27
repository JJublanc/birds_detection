from typing import Tuple

import git


def get_url_push(gitwd: str) -> str:
    """
    get distant push url from a git workdirectory
    :param gitwd: git working directory
    :return: url of the git remote used to push code
    """
    g = git.cmd.Git(gitwd)
    response = g.execute(["git", "remote", "show", "origin"])
    distant_url = response.split("Push  URL:")[1].split("\n")[0]
    return str(distant_url).strip()


def check_branch(branch: str, gitwd: str) -> Tuple[git.repo.base.Repo, str]:
    """
    Check if the current branch is the same as the one specified.
    Return the repo if it is.
    :param branch: branch name
    :param gitwd: working directory where .git folder is located
    :return:
    """
    distant_url = get_url_push(gitwd)
    repo = git.Repo(gitwd)
    local_branch = repo.active_branch.name
    try:
        assert local_branch == branch
    except AssertionError:
        raise AssertionError(
            f"You are not in branch {branch}. \n"
            f"Just checkout to this branch and try again!"
        )
    return repo, distant_url


def commit_code(repo: git.Repo, commit_msg: str) -> None:
    """
    Commit code to the repo
    :param repo: git repo
    :param commit_msg: message to commit
    :return: None
    """
    files = repo.git.diff(None, name_only=True)
    for f in files.split("\n"):
        repo.git.add(f)

    repo.git.commit("-m", commit_msg, "--no-verify")
