"""Adapter for any CSV already in the standard schema (also how you add another country)."""
import pandas as pd
from ..schema import validate


def load(path: str) -> pd.DataFrame:
    return validate(pd.read_csv(path))
