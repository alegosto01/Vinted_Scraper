# config/search_loader.py

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

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.wrong_words is None:
            self.wrong_words = []

def load_searches(path: str) -> Dict[str, SearchConfig]:
    """
    Loads searches from a YAML file and returns a dict:
    {search_name: SearchConfig}
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    searches: Dict[str, SearchConfig] = {}

    for name, data in raw.items():
        searches[name] = SearchConfig(**data)

    return searches
