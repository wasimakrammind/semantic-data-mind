import os
from typing import Union, BinaryIO

import pandas as pd
from loguru import logger

from src.ingestion.schema_detector import SchemaDetector
from src.ingestion.sampling import DataSampler
from src.models.dataset import Dataset, DatasetMetadata


class DataLoader:
    """Loads data from CSV, Excel, JSON, Parquet into a Dataset object."""

    SUPPORTED_FORMATS = {
        ".csv": "_load_csv",
        ".xlsx": "_load_excel",
        ".xls": "_load_excel",
        ".json": "_load_json",
        ".parquet": "_load_parquet",
    }

    def __init__(self):
        self.schema_detector = SchemaDetector()
        self.sampler = DataSampler()

    def load(self, source: Union[str, BinaryIO], filename: str) -> Dataset:
        """Load data from a file path or file-like object."""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{ext}'. Supported: {list(self.SUPPORTED_FORMATS.keys())}"
            )

        loader_method = getattr(self, self.SUPPORTED_FORMATS[ext])
        logger.info(f"Loading {filename} (format: {ext})")

        df = loader_method(source)
        logger.info(f"Loaded {len(df)} rows x {len(df.columns)} columns")

        metadata = self.schema_detector.detect(df, filename, ext)
        sample = self.sampler.create_llm_sample(df, metadata)
        profile = self.sampler.create_statistical_summary(df, metadata)

        return Dataset(
            df=df,
            metadata=metadata,
            sample_for_llm=sample,
            profile_summary=profile,
        )

    def _load_csv(self, source: Union[str, BinaryIO]) -> pd.DataFrame:
        return pd.read_csv(source, low_memory=False)

    def _load_excel(self, source: Union[str, BinaryIO]) -> pd.DataFrame:
        return pd.read_excel(source, engine="openpyxl")

    def _load_json(self, source: Union[str, BinaryIO]) -> pd.DataFrame:
        return pd.read_json(source)

    def _load_parquet(self, source: Union[str, BinaryIO]) -> pd.DataFrame:
        return pd.read_parquet(source)
