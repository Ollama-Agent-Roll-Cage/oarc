"""
Module: agent_access
Description:
    This module defines the AgentAccess data model using Pydantic's BaseModel.
    The AgentAccess class encapsulates a unique identifier for an agent,
    ensuring that the agent_id is provided and correctly validated.
Usage:
    Instantiate AgentAccess with a valid agent_id string to represent an agent's access details.
"""

from pydantic import BaseModel


class AgentAccess(BaseModel):
    """
    Represents the access details of an agent.

    Attributes:
        agent_id (str): A unique identifier for the agent. This field is required
                        and validated to ensure it is a valid string.
    """
    agent_id: str
   