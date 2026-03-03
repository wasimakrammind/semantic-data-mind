from typing import List, Optional

import pandas as pd
import json

from src.models.dataset import Dataset
from src.models.validation_result import ValidationFinding
from src.utils.serialization import dataframe_to_markdown


class SemanticValidator:
    """LLM-powered semantic validation that catches issues rule-based checks miss."""

    SYSTEM_PROMPT = """You are a data quality expert. Analyze the provided dataset sample and schema to identify semantic data quality issues that automated rules might miss.

Focus on:
1. Business logic violations (e.g., a deceased person with recent transactions)
2. Unit inconsistencies (e.g., mixing dollars and cents, km and miles)
3. Encoding issues (e.g., "N/A", "null", "-" used as values instead of proper nulls)
4. Domain-specific impossibilities (e.g., body temperature > 50°C, human age > 150)
5. Suspicious patterns (e.g., all values ending in 0, suspiciously round numbers)
6. Cross-column semantic issues that require domain understanding

Return your findings as a JSON array where each finding has:
- "severity": "error" | "warning" | "info"
- "column": column name or null
- "message": clear description of the issue
- "suggestion": recommended fix
- "evidence": supporting details as a dict

Return ONLY the JSON array, no other text."""

    def validate(
        self,
        dataset: Dataset,
        anthropic_client,
        max_sample_rows: int = 30,
    ) -> List[ValidationFinding]:
        sample = dataset.sample_for_llm
        if sample is None or len(sample) == 0:
            return []

        sample_md = dataframe_to_markdown(sample, max_rows=max_sample_rows)
        schema_info = self._format_schema(dataset)

        user_prompt = f"""Dataset Schema:
{schema_info}

Data Sample:
{sample_md}

Identify any semantic data quality issues in this dataset."""

        try:
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            response_text = response.content[0].text
            raw_findings = json.loads(response_text)

            findings = []
            for f in raw_findings:
                findings.append(
                    ValidationFinding(
                        severity=f.get("severity", "info"),
                        category="semantic",
                        column=f.get("column"),
                        row_indices=None,
                        message=f.get("message", ""),
                        suggestion=f.get("suggestion", ""),
                        evidence=f.get("evidence", {}),
                    )
                )
            return findings

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            return [
                ValidationFinding(
                    severity="info",
                    category="semantic",
                    column=None,
                    row_indices=None,
                    message=f"Semantic validation completed but could not parse LLM response: {e}",
                    suggestion="Try running semantic validation again.",
                    evidence={},
                )
            ]
        except Exception as e:
            return [
                ValidationFinding(
                    severity="info",
                    category="semantic",
                    column=None,
                    row_indices=None,
                    message=f"Semantic validation skipped: {e}",
                    suggestion="Check your Anthropic API key and try again.",
                    evidence={},
                )
            ]

    def _format_schema(self, dataset: Dataset) -> str:
        m = dataset.metadata
        lines = [f"Columns ({m.column_count}):"]
        for col in m.column_names:
            dtype = m.dtypes[col]
            missing = m.missing_summary.get(col, 0)
            line = f"  - {col} ({dtype})"
            if missing > 0:
                line += f" [{missing:.1f}% missing]"
            lines.append(line)
        return "\n".join(lines)
