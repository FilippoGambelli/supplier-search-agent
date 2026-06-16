class OrchestratorError(Exception):
    """Base exception for the agent_orchestrator module."""


class ArtifactError(OrchestratorError):
    """Error during artifact store operations."""


class ArtifactNotFoundError(ArtifactError):
    """Artifact ID not found in the store."""


class SubAgentError(OrchestratorError):
    """Error during sub-agent execution."""


class SearchAgentError(SubAgentError):
    """Web-search sub-agent returned an error."""


class DbManagerAgentError(SubAgentError):
    """Database manager sub-agent returned an error."""
