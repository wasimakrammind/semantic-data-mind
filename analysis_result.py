from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


@dataclass
class EDAResult:
    analysis_type: str  # "statistics", "correlation", "trend", "outlier", etc.
    columns_analyzed: List[str]
    results: Dict[str, Any]
    summary: str
    charts: List[Any] = field(default_factory=list)  # Plotly figure objects
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    text: str
    charts: List[Any] = field(default_factory=list)
    tool_calls_made: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
