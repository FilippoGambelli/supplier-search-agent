from typing import Any, Dict
import shortuuid
import time

class Artifact:
    def __init__(self, artifact_id: str, data: Any, meta: Dict = None):
        self.id = artifact_id
        self.data = data
        self.meta = meta or {}
        self.created_at = time.time()

class InMemoryArtifactStore:
    """
    Simple deterministic artifact store.
    """

    def __init__(self):
        self._store: Dict[str, Artifact] = {}

    def save(self, data: Any, meta: Dict = None) -> str:
        artifact_id = shortuuid.uuid()

        self._store[artifact_id] = Artifact(
            artifact_id=artifact_id,
            data=data,
            meta=meta
        )

        return artifact_id

    def load(self, artifact_id: str) -> Any:
        if artifact_id not in self._store:
            raise KeyError(f"Artifact not found: {artifact_id}")

        return self._store[artifact_id].data

    def load_full(self, artifact_id: str) -> Artifact:
        return self._store[artifact_id]

    def exists(self, artifact_id: str) -> bool:
        return artifact_id in self._store
    
artifact_store = InMemoryArtifactStore()