class DbManagerError(Exception):
    """Base exception for the agent_dbmanager module."""


class DatabaseError(DbManagerError):
    """Error during database operations."""


class IntegrityError(DatabaseError):
    """Constraint violation (duplicate PK, FK, UNIQUE, etc.)."""


class ConnectionError(DatabaseError):
    """Database connection error."""


class ArtifactError(DbManagerError):
    """Error during artifact store operations."""


class ArtifactNotFoundError(ArtifactError):
    """Artifact ID not found in the store."""


class InvalidJSONError(ArtifactError):
    """Artifact content is not valid JSON."""


class ValidationError(DbManagerError):
    """Input validation error (missing filters, invalid args)."""
