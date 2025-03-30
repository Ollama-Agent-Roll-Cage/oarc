"""
Module Description:
This module defines the ModelRequest class, a data model that represents the parameters 
required for a request. The ModelRequest class, built with Pydantic's BaseModel, enforces 
the data types for its attributes, ensuring that both model_name and agent_id are provided 
as strings.
"""

from pydantic import BaseModel


class ModelRequest(BaseModel):
    """
    Represents the parameters required for a model request, including the model name 
    and agent ID. Both fields are mandatory and must be provided as strings.
    """
    model_name: str
    agent_id: str
     