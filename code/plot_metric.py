import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("../derived_data/result.csv")
metric_list = ["accuracy", "recall", "precision"]

for m in metric_list:
    plt.plot(df["frac_spatial"], df[f"{m}_mean"], label=m)
    plt.fill_between(
        df["frac_spatial"],
        df[f"{m}_mean"] - df[f"{m}_std"],
        df[f"{m}_mean"] + df[f"{m}_std"],
        alpha=0.2,
    )

plt.xlabel("Fraction of spatial information")
plt.ylabel("metric value")

plt.legend(title="metrics")
plt.tight_layout()
plt.savefig("../figures/metric_chart.png")
