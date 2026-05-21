import time
import threading
from dataclasses import dataclass
from typing import Optional

_lock = threading.Lock()

@dataclass
class AgentStats:
    total_LLM_requests: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_execution_time: float = 0.0
    tool_calls: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def start(self):
        with _lock:
            self.start_time = time.time()

    def stop(self):
        with _lock:
            self.end_time = time.time()
            if self.start_time:
                self.total_execution_time = self.end_time - self.start_time

    def add_request(self, input_tokens: int = 0, output_tokens: int = 0):
        with _lock:
            self.total_LLM_requests += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_tokens += input_tokens + output_tokens

    def add_tool_call(self):
        with _lock:
            self.tool_calls += 1

    def to_dict(self):
        return {
            "total_LLM_requests": self.total_LLM_requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "total_execution_time_seconds": round(self.total_execution_time, 1),
            "tool_calls": self.tool_calls
        }


global_stats = AgentStats()

def get_stats() -> AgentStats:
    return global_stats

def reset_stats():
    with _lock:
        global_stats.total_LLM_requests = 0
        global_stats.total_tokens = 0
        global_stats.input_tokens = 0
        global_stats.output_tokens = 0
        global_stats.total_execution_time = 0.0
        global_stats.tool_calls = 0
        global_stats.start_time = None
        global_stats.end_time = None