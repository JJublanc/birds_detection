import json

from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication


def get_service_principal_auth(service_principal_file_name: str):
    """
    Get an service principal authentification object
    :param service_principal_file_name: json with the following fields :
    "appId", "displayName", "password" and "tenant"
    :return: service principal authentification object
    """
    with open(f"{service_principal_file_name}.json") as f:
        sp_config = json.load(f)

    sp = ServicePrincipalAuthentication(
        tenant_id=sp_config["tenant"],
        service_principal_id=sp_config["appId"],
        service_principal_password=sp_config["password"],
    )
    return sp


def get_workspace(
    workspace_config_file_name: str, sp: ServicePrincipalAuthentication
) -> Workspace:
    """
    Get a Workspace object from a json configuration
    :param workspace_config_file_name: json (that can been retrieve from CLI or
     Azure Portal)
    :param sp: Service Principal object
    :return: a Workspace
    """
    # Get workspace
    return Workspace.from_config(
        path=f"{workspace_config_file_name}.json", auth=sp
    )
