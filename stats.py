import time
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class AgentStats:
    total_LLM_requests: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_execution_time: float = 0.0
    tool_calls: int = 0
    errors: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        if self.start_time:
            self.total_execution_time = self.end_time - self.start_time

    def add_request(self, prompt_tokens: int = 0, generated_tokens: int = 0):
        self.total_LLM_requests += 1
        self.prompt_tokens += prompt_tokens
        self.generated_tokens += generated_tokens
        self.total_tokens += prompt_tokens + generated_tokens

    def add_tool_call(self):
        self.tool_calls += 1

    def add_error(self):
        self.errors += 1

    def to_dict(self):
        return {
            "total_LLM_requests": self.total_LLM_requests,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "total_execution_time_seconds": round(self.total_execution_time, 1),
            "tool_calls": self.tool_calls,
            "errors": self.errors
        }


_global_stats = AgentStats()

def get_stats() -> AgentStats:
    return _global_stats

def reset_stats():
    global _global_stats            # Global variable to hold the stats instance
    _global_stats = AgentStats()    # Reset the stats by creating a new instance