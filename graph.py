import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Ceas08_Enron.csv", low_memory=False)
df["Message"] = df["Message"].astype(str)
df["wc"] = df["Message"].str.split().str.len()
baseline = df["Category"].mean() * 100

# ---------- Figure 3.2: spam rate by message length ----------
buckets = [(0,5),(6,15),(16,50),(51,150),(151,500),(501,10**9)]
labels  = ["0–5","6–15","16–50","51–150","151–500","501+"]
rates, counts = [], []
for lo, hi in buckets:
    s = df[(df.wc >= lo) & (df.wc <= hi)]
    rates.append(s.Category.mean() * 100)
    counts.append(len(s))

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(labels, rates, color="#c44e52", edgecolor="black", linewidth=0.6)
ax.axhline(baseline, ls="--", color="#333", lw=1.2,
           label=f"Corpus baseline ({baseline:.1f}%)")
for b, r, n in zip(bars, rates, counts):
    ax.text(b.get_x()+b.get_width()/2, r+1.5, f"{r:.1f}%\n(n={n:,})",
            ha="center", fontsize=8)
ax.set_xlabel("Message length (words)")
ax.set_ylabel("Proportion labelled spam (%)")
ax.set_title("Spam rate by message length")
ax.set_ylim(0, 112); ax.legend()
plt.tight_layout(); plt.savefig("fig_length_confound.png", dpi=200)

# ---------- Figure 3.3: effect of merging Enron ----------
# replace 97.5 with your own recomputed CEAS-only value
fig, ax = plt.subplots(figsize=(5.5, 4.5))
bars = ax.bar(["CEAS_08 only", "CEAS_08 + Enron"], [97.5, 76.5],
              color=["#c44e52", "#55a868"], edgecolor="black", linewidth=0.6)
for b, v in zip(bars, [97.5, 76.5]):
    ax.text(b.get_x()+b.get_width()/2, v+1.5, f"{v:.1f}%", ha="center", fontweight="bold")
ax.set_ylabel("Short messages (≤15 words) labelled spam (%)")
ax.set_title("Effect of merging Enron on length confound")
ax.set_ylim(0, 110)
plt.tight_layout(); plt.savefig("fig_merge_effect.png", dpi=200)

# ---------- Figure 3.4: affective vocabulary ----------
terms = ["heart","love","my life","forever","beautiful","sweet","i love you"]
vals = []
low = df["Message"].str.lower()
for t in terms:
    m = df[low.str.contains(t, na=False, regex=False)]
    vals.append(m.Category.mean() * 100)

order = sorted(zip(terms, vals), key=lambda x: x[1])
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.barh([t for t,_ in order], [v for _,v in order],
        color="#4c72b0", edgecolor="black", linewidth=0.6)
ax.axvline(baseline, ls="--", color="#c44e52", lw=1.5,
           label=f"Corpus baseline ({baseline:.1f}%)")
for i,(t,v) in enumerate(order):
    ax.text(v+1, i, f"{v:.1f}%", va="center", fontsize=9)
ax.set_xlabel("Proportion of messages containing term labelled spam (%)")
ax.set_title("Spam association of affective vocabulary")
ax.set_xlim(0, 100); ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig("fig_vocab_bias.png", dpi=200)

print("Charts saved.")