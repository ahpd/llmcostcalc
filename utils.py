"""
Utility functions.
"""

from typing import Dict

import pandas as pd

from models import GoogleGeminiModel, Model, OpenAIModel, PricingInfo


def file_contents(file_path: str) -> str:
    """
    Read the contents of a file and return it as a string.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_models_info(csv_path: str) -> Dict[str, Model]:
    """
    Load model information from a CSV file. Header:

    model,model_full,vendor,rt_input,rt_cached_input,rt_output,batch_input,batch_output
    """

    df = pd.read_csv(csv_path)

    def parse_dollar(value):
        if isinstance(value, str) and value.startswith("$"):
            try:
                return float(value[1:])
            except ValueError:
                return None
        return None

    def get_pricing_info(row) -> PricingInfo:
        return PricingInfo(
            prompt_cost_realtime=parse_dollar(row["rt_input"]),
            prompt_cost_batch=parse_dollar(row["batch_input"]),
            completion_cost_realtime=parse_dollar(row["rt_output"]),
            completion_cost_batch=parse_dollar(row["batch_output"]),
        )

    models_info = {}

    for _, row in df.iterrows():
        vendor = row["vendor"].lower()
        model_id = row["model"].lower()
        name = row["model_full"]
        tokenizer_model = row["tokenizer_model"]
        pricing_info = get_pricing_info(row)

        if vendor == "openai":
            model = OpenAIModel(
                model_id=model_id,
                name=name,
                tokenizer_model=tokenizer_model,
                pricing=pricing_info,
            )
        elif vendor == "google":
            model = GoogleGeminiModel(
                model_id=model_id,
                name=name,
                tokenizer_model=tokenizer_model,
                pricing=pricing_info,
            )
        else:
            raise ValueError(f"Unknown vendor: {vendor}")

        models_info[model_id] = model

    return models_info
