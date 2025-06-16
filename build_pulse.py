import pandas as pd
import numpy as np
from textblob import TextBlob
import argparse
import os
import re

# Define keywords/phrases commonly linked to unpaid opportunities
UNPAID_KEYWORDS = [
    r"\bunpaid\b",
    r"\bno\s+pay\b",
    r"\bno\s+compensation\b",
    r"\bvolunteer\b",
    r"\bexperience\s+only\b",
]

# Regular expression to catch numeric values like 15, 100, etc.
RATE_VALUE_REGEX = re.compile(r"\b(\d{1,5})\b")


def get_rate(description):
    if pd.isnull(description):
        return None

    content = str(description).lower().strip()

    # Check if any unpaid keyword is present
    if any(re.search(keyword, content) for keyword in UNPAID_KEYWORDS):
        return None

    # Extract the first valid numeric rate found
    rate_match = RATE_VALUE_REGEX.search(content)
    return int(rate_match.group(1)) if rate_match else None


def map_region_code(work_location):
    if isinstance(work_location, str):
        work_location = work_location.lower()
        if "los angeles" in work_location or "la" in work_location:
            return "LA"
        elif "new york" in work_location or "ny" in work_location:
            return "NY"
        elif "chicago" in work_location or "il" in work_location:
            return "CHI"
        elif "atlanta" in work_location or "ga" in work_location:
            return "ATL"
        elif "london" in work_location:
            return "LON"
        elif "mexico" in work_location:
            return "MEX"
    return "Other"


def map_proj_type_code(project_type):
    if isinstance(project_type, str):
        project_type = project_type.lower()
        if "film" in project_type:
            return "F"
        elif "series" in project_type or "tv" in project_type:
            return "T"
        elif "commercial" in project_type:
            return "C"
        elif "voice" in project_type:
            return "V"
    return "V"


def round_to_nearest_25(rate):
    if pd.isna(rate):
        return None
    return int(round(rate / 25.0)) * 25


def get_sentiment(text):
    if isinstance(text, str):
        return TextBlob(text).sentiment.polarity
    return None


def round_to_nearest_0_05(score):
    if pd.isna(score):
        return None
    return round(score * 20) / 20.0


def has_ai_theme(text):
    if isinstance(text, str):
        text = text.lower()
        return any(keyword in text for keyword in ["ai", "robot", "android"])
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Casting Pulse summary.")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input)
    df["date_utc"] = pd.to_datetime(df["posted_date"]).dt.strftime("%Y-%m-%d")
    df["region_code"] = df["work_location"].map(map_region_code)
    df["proj_type_code"] = df["project_type"].map(map_proj_type_code)

    df.rename(columns={"rate": "raw_rate_text"}, inplace=True)
    # ✅ Apply the extract_first_valid_rate to raw rate text column
    df["rate"] = df["raw_rate_text"].apply(get_rate)

    df["sentiment"] = df["role_description"].apply(get_sentiment)
    df["has_ai_theme"] = df["role_description"].apply(has_ai_theme)

    aggregated_df = (
        df.groupby(["date_utc", "region_code", "proj_type_code"])
        .agg(
            role_count_day=("id", "count"),
            lead_principal_count=(
                "role_type",
                lambda x: x.str.contains("Lead|Principal", na=False).sum(),
            ),
            union_count=("union", lambda x: x.str.contains("Union", na=False).sum()),
            median_rate_day_usd=(
                "rate",
                lambda x: round_to_nearest_25(
                    pd.to_numeric(x, errors="coerce").median()
                ),
            ),
            sentiment_mean=("sentiment", "mean"),
            ai_theme_count=("has_ai_theme", "sum"),
        )
        .reset_index()
    )

    aggregated_df["lead_share_pct_day"] = round(
        (aggregated_df["lead_principal_count"] / aggregated_df["role_count_day"]) * 100,
        1,
    )
    aggregated_df["union_share_pct_day"] = round(
        (aggregated_df["union_count"] / aggregated_df["role_count_day"]) * 100, 1
    )
    aggregated_df["sentiment_avg_day"] = aggregated_df["sentiment_mean"].apply(
        round_to_nearest_0_05
    )
    aggregated_df["theme_ai_share_pct_day"] = round(
        (aggregated_df["ai_theme_count"] / aggregated_df["role_count_day"]) * 100, 1
    )

    aggregated_df.drop(
        columns=[
            "lead_principal_count",
            "union_count",
            "sentiment_mean",
            "ai_theme_count",
        ],
        inplace=True,
    )

    filtered_df = aggregated_df[aggregated_df["role_count_day"] >= 5].copy()

    sensitive_cols = [
        "role_count_day",
        "lead_share_pct_day",
        "union_share_pct_day",
        "median_rate_day_usd",
        "sentiment_avg_day",
        "theme_ai_share_pct_day",
    ]

    np.random.seed(42)
    for col in sensitive_cols:
        if col in filtered_df.columns and pd.api.types.is_numeric_dtype(
            filtered_df[col]
        ):
            filtered_df[col] = filtered_df[col].apply(
                lambda x: x + np.random.laplace(0, 1) if pd.notna(x) else x
            )

    sorted_df = filtered_df.sort_values(
        by=["date_utc", "region_code", "proj_type_code"]
    )

    # Ensure the output directory exists
    output_path = args.output

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    sorted_df.to_csv(args.output, index=False)
    print(f"Data saved to {args.output}")


if __name__ == "__main__":
    main()
