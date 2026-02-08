"""
Aggregate paper-to-slide generation metrics from per-paper JSON files
and report outliers with paper + slide IDs.

Usage:
    python aggregate_generation_metrics.py /path/to/json_folder
"""

import os
import json
import math
import argparse
from collections import defaultdict

import pandas as pd

# -------------------------
# Utilities
# -------------------------

def is_valid_number(v):
    return isinstance(v, (int, float)) and not math.isnan(v)


def safe_mean(values):
    clean = [v for v in values if is_valid_number(v)]
    return sum(clean) / len(clean) if clean else None


def extract_layout_quality(layout):
    """
    Layout quality = fraction of elements that are inside canvas.
    """
    if not layout or "element_in_canvas" not in layout:
        return None
    elems = layout["element_in_canvas"].values()
    if not elems:
        return None
    return sum(bool(v) for v in elems) / len(elems)


# -------------------------
# Outlier detection (IQR)
# -------------------------

def find_outliers(entries, k=1.5):
    """
    entries: list of dicts with key 'value'
    returns: (low_outliers, high_outliers)
    """
    clean = [e for e in entries if is_valid_number(e["value"])]
    if len(clean) < 4:
        return [], []

    values = sorted(e["value"] for e in clean)
    q1 = values[len(values) // 4]
    q3 = values[(3 * len(values)) // 4]
    iqr = q3 - q1

    low_thresh = q1 - k * iqr
    high_thresh = q3 + k * iqr

    low = [e for e in clean if e["value"] < low_thresh]
    high = [e for e in clean if e["value"] > high_thresh]
    return low, high


# -------------------------
# JSON processing
# -------------------------

def process_json_file(path, accum):
    paper_id = os.path.splitext(os.path.basename(path))[0]

    with open(path, "r") as f:
        data = json.load(f)

    # -------- Paper-level metrics --------
    paper_metrics = [
        ("deck_dtw", "DTW"),
        ("deck_transition", "TransSim"),
        ("deck_fidelity", "DeckFid"),
        ("deck_faithfulness", "DeckFaith"),
        ("deck_perplexity", "DeckPPL")
    ]

    for json_key, metric_name in paper_metrics:
        accum[metric_name].append({
            "value": data.get(json_key),
            "paper_id": paper_id,
            "slide_id": None
        })

    # # Coherence proxy (higher is better)
    # deck_perplexity = data.get("deck_perplexity")
    # # coherence = (1.0 / deck_perplexity) if deck_perplexity and deck_perplexity > 0 else None
    # accum["DeckPPL"].append({
    #     "value": deck_perplexity,
    #     "paper_id": paper_id,
    #     "slide_id": None
    # })

    # -------- Slide-level metrics --------
    for slide_id, m in enumerate(data.get("matches", [])):
        slide_metrics = [
            ("perplexity", "PPL"),
            ("text_similarity", "TextSim"),
            ("faithfulness", "Faith."),
            ("figure_similarity", "FigAlign"),
        ]

        for json_key, metric_name in slide_metrics:
            accum[metric_name].append({
                "value": m.get(json_key),
                "paper_id": paper_id,
                "slide_id": slide_id
            })

        layout_q = extract_layout_quality(m.get("layout"))
        accum["LayoutQ"].append({
            "value": layout_q,
            "paper_id": paper_id,
            "slide_id": slide_id
        })


# -------------------------
# Aggregation + reporting
# -------------------------

def aggregate_metric(entries):
    return safe_mean([e["value"] for e in entries])


def analyze_folder(folder, output_folder="", save_output=False, report_outliers=False):

    if not os.path.exists(folder):
        print('Error: no evaluation folder found: ', folder)
        return {}

    accum = defaultdict(list)

    num_files = 0
    for fname in os.listdir(folder):
        if fname.endswith(".json"):
            process_json_file(os.path.join(folder, fname), accum)
            num_files+=1

    metrics = {k: aggregate_metric(v) for k, v in accum.items()}

    print('')
    print (f"Number of evaluated decks: {num_files}")

    if report_outliers:
        print("\n================ OUTLIER REPORT ================")

        for metric, entries in accum.items():
            low, high = find_outliers(entries)

            nans = [
                e for e in entries
                if e["value"] is None or
                (isinstance(e["value"], float) and math.isnan(e["value"]))
            ]

            if not (low or high or nans):
                continue

            print(f"\n--- {metric} ---")

            for e in low:
                print(
                    f"LOW   {e['value']:.4f} | paper={e['paper_id']} slide={e['slide_id']}"
                )

            for e in high:
                print(
                    f"HIGH  {e['value']:.4f} | paper={e['paper_id']} slide={e['slide_id']}"
                )

            for e in nans:
                print(
                    f"NaN   --      | paper={e['paper_id']} slide={e['slide_id']}"
                )


    # Show results
    slide_rows = []
    slide_row = {
        'PPL': metrics.get('PPL'),
        'TextSim': metrics.get('TextSim'),
        'Faith': metrics.get('Faith.'),
        'FigAlign': metrics.get('FigAlign'),
        'LayoutQ': metrics.get('LayoutQ')
    }
    slide_rows.append(slide_row)
    df_slide = pd.DataFrame(slide_rows)

    deck_rows = []
    deck_row = {
        'DeckPPL': metrics.get('DeckPPL'),
        'DeckFaith': metrics.get('DeckFaith'),
        'DeckFid': metrics.get('DeckFid'),
        'DTW': metrics.get('DTW'),
        'TransSim': metrics.get('TransSim')
    }
    deck_rows.append(deck_row)
    df_deck = pd.DataFrame(deck_rows)

    print("\nSlide-level Metrics:")
    print(df_slide)
    print('')

    print("\nDeck-level Metrics:")
    print(df_deck)
    print('')

    # Save results
    if save_output:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        merged_df = pd.concat([df_slide, df_deck], axis=1)
        # print(merged_df)

        output_file_path = os.path.join(output_folder, 'generation_metrics.csv' )
        merged_df.to_csv(output_file_path, index=False)
        print(f"Saved generation evaluation metrics to {output_file_path}")

    return metrics


# -------------------------
# LaTeX table printing
# -------------------------

def print_latex_row(metrics):

    def fmt(x):
        return f"{x:.3f}" if x is not None else "--"



    title_row = (
        "PPL &"
        " TextSim &"
        " Faith &"
        " FigAlign &"
        " LayoutQ \\\\"
    )
    row = (
        f"{fmt(metrics.get('PPL'))} & "
        f"{fmt(metrics.get('TextSim'))} & "
        f"{fmt(metrics.get('Faith.'))} & "
        f"{fmt(metrics.get('FigAlign'))} & "
        f"{fmt(metrics.get('LayoutQ'))} \\\\"
    )
    print("\nSlide-level Metrics:")
    print(title_row)
    print(row)




    deck_title_row = (
        "DeckPPL &"
        " DeckFaith &"
        " DeckFid &"
        " DTW &"
        " TransSim \\\\"
    )
    row = (
    f"{fmt(metrics.get('DeckPPL'))}  &  "
    f"{fmt(metrics.get('DeckFaith'))}  &  "
    f"{fmt(metrics.get('DeckFid'))}  &  "
    f"{fmt(metrics.get('DTW'))}  &  "
    f"{fmt(metrics.get('TransSim'))} \\\\"
    )
    print("\nDeck-level Metrics:")
    print(deck_title_row)
    print(row)


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_folder", required=True, help="Folder containing per-paper JSON files")
    parser.add_argument("--output_folder", default="./", help="Analysis output folder path")
    parser.add_argument('--save_output', action='store_true', help='vesave output tto output folder')
    # parser.add_argument("--model", default="ModelName", help="Model name for LaTeX table")
    # parser.add_argument("--no-outliers", action="store_true", help="Disable outlier reporting")
    args = parser.parse_args()

    metrics = analyze_folder(args.evaluation_folder, args.output_folder, args.save_output)
    # print_latex_row(metrics)
