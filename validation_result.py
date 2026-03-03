from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime


@dataclass
class ValidationFinding:
    severity: str  # "error", "warning", "info"
    category: str  # "statistical", "temporal", "schema", "cross_column", "semantic"
    column: Optional[str]
    row_indices: Optional[List[int]]
    message: str
    suggestion: str
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    findings: List[ValidationFinding]
    summary: str
    data_quality_score: float  # 0-100
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "warning")

    @property
    def info_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "info")
