from dataclasses import dataclass
from typing import Dict, List

import yaml


@dataclass
class SearchConfig:
    search: str
    prezzoDa: str = ""
    prezzoA: str = ""
    condition: str = ""
    colore: str = ""
    brands: str = ""
    sort: str = "newest_first"
    category: str = ""
    folder: str = ""
    enabled: bool = True
    tags: List[str] = None
    wrong_words: List[str] = None
    auto_schedule: bool = True
    min_delay_minutes: int = 15
    max_delay_minutes: int = 120
    activity_window_hours: int = 24
    target_items_per_run: float = 6.0
    sold_weight: float = 2.0
    default_delay_minutes: int = 60
    target_value_per_run: float = 4.0
    good_deal_weight: float = 1.0
    fast_sale_weight: float = 1.5
    fast_sale_hours: float = 12.0
    good_deal_min_score: float = 2.0
    good_deal_min_confidence: float = 0.6
    no_residential: bool = False
    cascade_only: bool = False

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.wrong_words is None:
            self.wrong_words = []


def load_searches(path: str) -> Dict[str, SearchConfig]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return {}

    if not isinstance(raw, dict):
        raise ValueError(f"Searches YAML must contain a mapping of search names to configs: {path}")

    searches: Dict[str, SearchConfig] = {}

    for name, data in raw.items():
        if not isinstance(data, dict):
            raise ValueError(f"Search entry '{name}' must be a mapping, got {type(data).__name__}")
        searches[name] = SearchConfig(**data)

    return searches
