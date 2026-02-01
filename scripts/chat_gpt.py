import os
import csv
import re
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

JSON_SCHEMA = {
    "name": "negative_keywords",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "negatives": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "keyword": {"type": "string"},
                        "type": {
                            "type": "string",
                            "enum": [
                                "accessory",
                                "wrong_model",
                                "wrong_version",
                                "digital",
                                "bundle",
                                "packaging",
                                "manual",
                                "parts_repair",
                                "other"
                            ]
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "notes": {"type": "string"}
                    },
                    "required": ["keyword", "type", "confidence", "notes"]
                },
                "minItems": 10,
                "maxItems": 120
            }
        },
        "required": ["negatives"]
    }
}

def safe_filename(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

def generate_negative_keywords_csv(
    search_query: str,
    output_dir: str = ".",
    locale: str = "it-IT"
) -> str:
    """
    Generates a CSV file with negative keywords for a search query.
    Returns the path to the CSV.
    """

    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "system",
                "content": (
                    "You generate negative keywords to exclude irrelevant Vinted listings by title. "
                    "You must be conservative: never include brand, model, version, platform, size, "
                    "capacity or any identity-defining token from the query. "
                    "Prefer short keywords or short phrases (1–3 words). "
                    "Include multilingual variants commonly seen on EU marketplaces (IT/ES/EN) when relevant."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Search query: {search_query}\n"
                    f"Marketplace: Vinted\n"
                    f"Locale: {locale}\n\n"
                    "Generate negative keywords that would filter out:\n"
                    "- accessories\n"
                    "- wrong models or versions\n"
                    "- digital items (codes, accounts, DLC)\n"
                    "- packaging-only listings\n"
                    "- parts, broken, repair items\n"
                    "- bundles or sets if not the main product\n"
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "json_schema": JSON_SCHEMA
            }
        }
    )

    data = response.output_text  # parsed JSON dict

    filename = f"negatives_{safe_filename(search_query)}.csv"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["keyword", "type", "confidence", "notes"])

        for item in data["negatives"]:
            writer.writerow([
                item["keyword"].lower().strip(),
                item["type"],
                round(item["confidence"], 2),
                item["notes"]
            ])

    return filepath
