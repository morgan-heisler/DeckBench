import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Configuration
# ----------------------------
JSON_FOLDER = "/root/data/daesik/multiturn_eval_executive_pdf"  # change to your folder
OUTPUT_CSV = "executive_deck_metrics_summary.csv"
OUTPUT_PLOT = "executive_deck_metrics_per_turn.png"
OUTPUT_DELTA_PLOT = "executive_deck_metrics_delta_per_turn.png"
OUTPUT_RATE_PLOT = "executive_deck_rate_change_per_turn.png"
RELATIVE_RATE_CSV = "executive_baseline_relative_rate_metrics.csv"
RELATIVE_RATE_SUMMARY = "executive_baseline_relative_rate_summary.csv"
RELATIVE_RATE_PLOT = "executive_baseline_relative_rate_vs_turn.png"

# ----------------------------
# Step 1: Read all JSONs
# ----------------------------
all_data = []

for root, dirs, files in os.walk(JSON_FOLDER):
    for file in files:
        if file.endswith(".json"):
            path = os.path.join(root, file)
            with open(path, "r") as f:
                data = json.load(f)
                # Each key is a turn (0,1,2,...)
                for turn_key, metrics in data.items():
                    # Extract number from keys like "0", "1", or "Turn1"
                    import re
                    match = re.search(r'\d+', turn_key)
                    if match:
                        turn = int(match.group())
                    else:
                        continue  # skip keys that don't have a number
                    row = {
                        "file": file,
                        "turn": int(turn),
                        "deck_dtw": metrics.get("deck_dtw"),
                        "deck_transition": metrics.get("deck_transition"),
                        "deck_perplexity": metrics.get("deck_perplexity"),
                        "deck_faithfulness": metrics.get("deck_faithfulness"),
                        "deck_fidelity": metrics.get("deck_fidelity")
                    }
                    # Optional: compute average slide-level metrics
                    slide_matches = metrics.get("slide_matches", [])
                    if slide_matches:
                        text_sims = [s.get("text_similarity") for s in slide_matches if s.get("text_similarity") is not None]
                        figure_sims = [s.get("figure_similarity") for s in slide_matches if s.get("figure_similarity") is not None]
                        faithfulness = [s.get("faithfulness") for s in slide_matches if s.get("faithfulness") is not None]
                        perplexities = [s.get("perplexity") for s in slide_matches if s.get("perplexity") is not None]

                        row["avg_text_similarity"] = np.mean(text_sims) if text_sims else None
                        row["avg_figure_similarity"] = np.mean(figure_sims) if figure_sims else None
                        row["avg_slide_faithfulness"] = np.mean(faithfulness) if faithfulness else None
                        row["avg_perplexity"] = np.mean(perplexities) if perplexities else None
                    
                    # Rate-change metrics (only available for turn 1+)
                    rate_change = metrics.get("ref_based_rate_change", {})
                    row["dtw_rate_change"] = rate_change.get("dtw_rate_change")
                    row["transition_rate_change"] = rate_change.get("transition_rate_change")

                    all_data.append(row)

df = pd.DataFrame(all_data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved aggregated metrics to {OUTPUT_CSV}")

# ----------------------------
# Step 2: Analysis of metric trends across turns
# ----------------------------
metrics_to_plot = [
    "deck_dtw", "deck_transition", "deck_perplexity", 
    "deck_faithfulness", "deck_fidelity",
    "avg_text_similarity", "avg_figure_similarity", "avg_slide_faithfulness", "avg_perplexity"
]

summary = df.groupby("turn")[metrics_to_plot].agg(["mean", "std"])
print("Summary statistics per turn:")
print(summary)

# Plot metrics over editing turns
plt.figure(figsize=(15, 10))
for i, metric in enumerate(metrics_to_plot, 1):
    plt.subplot(3, 3, i)
    grouped = df.groupby("turn")[metric]
    means = grouped.mean()
    stds = grouped.std()
    plt.errorbar(
        means.index, means.values, yerr=stds.values,
        fmt='-o', capsize=4
    )
    plt.title(metric)
    plt.xlabel("Editing Turn")
    plt.ylabel(metric)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
print(f"Saved metrics plot to {OUTPUT_PLOT}")

# ----------------------------
# Step 3: Compute Delta / Change per turn
# ----------------------------
delta_summary = summary.copy()
for metric in metrics_to_plot:
    delta_summary[(metric, 'delta')] = summary[(metric, 'mean')].diff().fillna(0)  # turn-over-turn change

# Optional: save delta table
delta_summary.to_csv("deck_metrics_delta_summary.csv")
print("Saved delta per turn to deck_metrics_delta_summary.csv")

# ----------------------------
# Step 4: Plot Delta / Change per turn
# ----------------------------
plt.figure(figsize=(15, 10))
for i, metric in enumerate(metrics_to_plot, 1):
    plt.subplot(3, 3, i)
    delta = delta_summary[(metric, 'delta')]
    plt.bar(delta.index, delta.values)
    plt.title(f"{metric} Δ per turn")
    plt.xlabel("Editing Turn")
    plt.ylabel("Change")
plt.tight_layout()
plt.savefig(OUTPUT_DELTA_PLOT)
print(f"Saved delta metrics plot to {OUTPUT_DELTA_PLOT}")

# ----------------------------
# Step 5: Plot Rate-Change metrics (dtw_rate_change, transition_rate_change)
# ----------------------------
rate_metrics = ["dtw_rate_change", "transition_rate_change"]
plt.figure(figsize=(10, 5))
for metric in rate_metrics:
    grouped = df.groupby("turn")[metric].mean()
    plt.plot(grouped.index, grouped.values, marker='o', label=metric)
plt.title("Rate-Change Metrics per Turn")
plt.xlabel("Editing Turn")
plt.ylabel("Rate Change")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_RATE_PLOT)
print(f"Saved rate-change metrics plot to {OUTPUT_RATE_PLOT}")


# ----------------------------
# Step A: Compute baseline-relative rate (w.r.t turn 0)
# ----------------------------

rate_rows = []

for file, group in df.groupby("file"):
    group = group.sort_values("turn")

    # Require turn 0 to exist
    if 0 not in group["turn"].values:
        continue

    base = group[group["turn"] == 0].iloc[0]
    base_dtw = base["deck_dtw"]
    print(base_dtw)
    base_trans = base["deck_transition"]

    for _, row in group.iterrows():
        turn = row["turn"]

        if turn == 0:
            continue

        rate_rows.append({
            "file": file,
            "turn": turn,
            "dtw_abs_change_from_0": row["deck_dtw"] - base_dtw,
            "transition_abs_change_from_0": row["deck_transition"] - base_trans,
            "dtw_rate_from_0": (
                (row["deck_dtw"] - base_dtw) / abs(base_dtw)
                if base_dtw is not None else None
            ),
            "transition_rate_from_0": (
                (row["deck_transition"] - base_trans) / abs(base_trans)
                if base_trans is not None else None
            ),
        })

rate_df = pd.DataFrame(rate_rows)
rate_df.to_csv(RELATIVE_RATE_CSV, index=False)
print(f"Saved baseline-relative rate metrics to {RELATIVE_RATE_CSV}")


rate_summary = rate_df.groupby("turn").agg(
    mean_dtw_rate=("dtw_rate_from_0", "mean"),
    std_dtw_rate=("dtw_rate_from_0", "std"),
    mean_trans_rate=("transition_rate_from_0", "mean"),
    std_trans_rate=("transition_rate_from_0", "std"),
)

rate_summary.to_csv(RELATIVE_RATE_SUMMARY)
print(f"Saved baseline-relative rate summary to {RELATIVE_RATE_SUMMARY}")

# plt.figure(figsize=(8, 5))

# plt.errorbar(
#     rate_summary.index,
#     rate_summary["mean_dtw_rate"],
#     # yerr=rate_summary["std_dtw_rate"],
#     marker="o",
#     label="DTW rate vs turn 0"
# )

# plt.errorbar(
#     rate_summary.index,
#     rate_summary["mean_trans_rate"],
#     # yerr=rate_summary["std_trans_rate"],
#     marker="o",
#     label="Transition similarity rate vs turn 0"
# )

# plt.axhline(0, linestyle="--", linewidth=1)
# plt.xlabel("Editing Turn")
# plt.ylabel("Relative Change vs Turn 0")
# plt.title("Baseline-relative Improvement over Editing Turns")
# plt.legend()
# plt.tight_layout()
# plt.savefig("baseline_relative_rate_vs_turn.png")
# print("Saved baseline-relative rate plot to baseline_relative_rate_vs_turn.png")

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5, 4))  # smaller for half-page figure

# Original x and y
x_orig = rate_summary.index.astype(int)

# Prepend turn 0
x = np.insert(x_orig, 0, 0)
y_dtw = np.insert(rate_summary["mean_dtw_rate"].values, 0, 0.0)
y_trans = np.insert(rate_summary["mean_trans_rate"].values, 0, 0.0)

# Plot DTW rate
plt.plot(
    x,
    y_dtw,
    marker="o",
    linewidth=2,
    markersize=6,
    label="DTW rate vs turn 0"
)

# Plot transition similarity rate
plt.plot(
    x,
    y_trans,
    marker="o",
    linewidth=2,
    markersize=6,
    label="Transition similarity rate vs turn 0"
)

# Horizontal line at 0 (baseline)
plt.axhline(0, linestyle="--", linewidth=1.5, color="gray")

plt.xlabel("Editing Turn")
plt.ylabel("Change vs Turn 0")
plt.title("Baseline-relative Improvement over Editing Turns", fontsize=10)

# Only whole numbers on x-axis
plt.xticks(x)


# Axis labels and title
plt.xlabel("Editing Turn")
plt.ylabel("Relative Change vs Turn 0")
plt.title("Baseline-relative Improvement over Editing Turns", fontsize=10)

# Move legend outside the plot (right side)
# plt.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.legend(fontsize=9)
plt.tight_layout()

# Save figure
plt.savefig(RELATIVE_RATE_PLOT, dpi=300)
print(f"Saved baseline-relative rate plot to {RELATIVE_RATE_PLOT}")


