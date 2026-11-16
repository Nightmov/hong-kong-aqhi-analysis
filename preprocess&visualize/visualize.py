import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

TYPE_COLOR = {"Roadside": "#c0392b", "Urban": "#2c7fb8", "Rural": "#27ae60"}
TYPE_ORDER = ["Roadside", "Urban", "Rural"]


def load():
    df = pd.read_csv(os.path.join(OUT, "aqhi_long.csv"),
                     usecols=["datetime", "date", "hour", "year", "month",
                              "weekday", "is_weekend", "season", "station",
                              "aqhi", "is_estimated", "is_capped"],
                     parse_dates=["datetime"])
    st = pd.read_csv(os.path.join(OUT, "stations.csv"))
    # Measured only: drop estimated values and missing rows
    m = df[(df["is_estimated"] == 0) & df["aqhi"].notna()].copy()
    m = m.merge(st[["station_code", "station_en", "station_type"]],
                left_on="station", right_on="station_code", how="left")
    return m, st


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p)
    plt.close(fig)
    print("  saved", name)


# fig1 Diurnal profile
def fig1(m):
    hourly = m.groupby("hour")["aqhi"].mean()
    picks = ["Causeway Bay", "Sha Tin", "Tap Mun"]
    labels = {"Causeway Bay": "Causeway Bay (roadside)",
              "Sha Tin": "Sha Tin (urban)",
              "Tap Mun": "Tap Mun (rural)"}
    colors = {"Causeway Bay": TYPE_COLOR["Roadside"],
              "Sha Tin": TYPE_COLOR["Urban"],
              "Tap Mun": TYPE_COLOR["Rural"]}

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(hourly.index, hourly.values, color="black", lw=2.2,
            marker="o", ms=5, label="All 18 stations", zorder=5)
    for s in picks:
        v = m[m["station"] == s].groupby("hour")["aqhi"].mean()
        ax.plot(v.index, v.values, lw=1.5, color=colors[s], alpha=0.9,
                marker="", ls="--", label=labels[s])

    lo, hi = hourly.idxmin(), hourly.idxmax()
    ax.scatter([lo], [hourly[lo]], s=70, facecolor="white",
               edgecolor="black", zorder=6)
    ax.annotate(f"Lowest {hourly[lo]:.2f} at hour {lo:02d}",
                xy=(lo, hourly[lo]), xytext=(lo + 1.4, hourly[lo] - 0.35),
                fontsize=9.5, arrowprops=dict(arrowstyle="->", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.6"))
    ax.annotate(f"Highest {hourly[hi]:.2f} at hour {hi:02d}",
                xy=(hi, hourly[hi]), xytext=(hi - 6.2, hourly[hi] + 0.28),
                fontsize=9.5, arrowprops=dict(arrowstyle="->", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.6"))

    ax.set_xlabel("Hour of day (EPD label 01-24; label 24 = 23:00-24:00)")
    ax.set_ylabel("Mean AQHI")
    ax.set_title("Fig. 1  Average diurnal profile of AQHI, 2013-2026")
    ax.set_xticks(range(1, 25))
    ax.set_xlim(0.6, 24.4)
    ax.legend(loc="upper left", frameon=False, ncol=1)
    ax.text(0.99, -0.19, f"Amplitude between lowest and highest hour: "
                         f"{hourly[hi]-hourly[lo]:.2f} AQHI "
                         f"({(hourly[hi]-hourly[lo])/hourly.mean()*100:.0f}% of the mean)",
            transform=ax.transAxes, ha="right", fontsize=9, color="0.25")
    save(fig, "fig1_hourly_profile.png")
    return hourly


# fig2 Station x hour heatmap
def fig2(m):
    mat = m.pivot_table(index="station", columns="hour", values="aqhi", aggfunc="mean")
    mat = mat.loc[mat.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(9.6, 8.2))
    im = ax.imshow(mat.values, aspect="auto", cmap="YlOrRd", vmin=2, vmax=5)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([f"{c:02d}" for c in mat.columns], fontsize=8.5)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=9)
    ax.set_xlabel("Hour of day (EPD label 01-24)")
    ax.set_ylabel("Station (ordered by mean AQHI)")
    ax.set_title("Fig. 2  Mean AQHI by station and hour of day")
    ax.grid(False)

    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("Mean AQHI")

    hi_pos = np.unravel_index(np.nanargmax(mat.values), mat.shape)
    lo_pos = np.unravel_index(np.nanargmin(mat.values), mat.shape)
    for pos, ls in ((hi_pos, "-"), (lo_pos, "--")):
        ax.add_patch(plt.Rectangle((pos[1] - .5, pos[0] - .5), 1, 1, fill=False,
                                   ec="black", lw=1.6, ls=ls))
    fig.text(0.02, -0.015,
             f"Solid box: highest cell {mat.values[hi_pos]:.2f} "
             f"({mat.index[hi_pos[0]]}, hour {mat.columns[hi_pos[1]]:02d});   "
             f"dashed box: lowest cell {mat.values[lo_pos]:.2f} "
             f"({mat.index[lo_pos[0]]}, hour {mat.columns[lo_pos[1]]:02d}). "
             f"Measured values only.",
             fontsize=8.5, color="0.25")
    save(fig, "fig2_station_hour_heatmap.png")
    return mat


# fig3 Station ranking and extreme share
def fig3(m):
    g = m.groupby(["station", "station_type"]).agg(
        mean_aqhi=("aqhi", "mean"),
        high_pct=("aqhi", lambda s: (s >= 7).mean() * 100),
        n=("aqhi", "size")).reset_index()
    g = g.sort_values("mean_aqhi")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.4),
                             gridspec_kw={"width_ratios": [3, 1.15], "wspace": 0.05})
    ax = axes[0]
    cols = [TYPE_COLOR[t] for t in g["station_type"]]
    ax.barh(g["station"], g["mean_aqhi"], color=cols, edgecolor="white", height=0.72)
    for i, (v, n) in enumerate(zip(g["mean_aqhi"], g["n"])):
        ax.text(v + 0.015, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlabel("Mean AQHI (measured values only)")
    ax.set_title("Fig. 3  Stations ranked by mean AQHI")
    ax.set_xlim(2.9, g["mean_aqhi"].max() + 0.18)
    handles = [plt.Rectangle((0, 0), 1, 1, color=TYPE_COLOR[t]) for t in TYPE_ORDER]
    ax.legend(handles, TYPE_ORDER, loc="lower right", frameon=False, title="Station type")

    ax2 = axes[1]
    ax2.barh(g["station"], g["high_pct"], color="#7f7f7f", height=0.72)
    for i, v in enumerate(g["high_pct"]):
        ax2.text(v + 0.08, i, f"{v:.1f}%", va="center", fontsize=9)
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([])
    ax2.set_xlabel("Share of hours with AQHI >= 7 (%)")
    ax2.set_title("High-pollution frequency", fontsize=11)
    ax2.set_xlim(0, g["high_pct"].max() * 1.35)
    save(fig, "fig3_station_ranking.png")
    return g


# fig4 Annual trend
def fig4(m):
    d = m[m["year"] >= 2014]
    y = d.groupby("year").agg(mean_aqhi=("aqhi", "mean"),
                              high_pct=("aqhi", lambda s: (s >= 7).mean() * 100),
                              n=("aqhi", "size")).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5.4))
    bars = ax.bar(y["year"], y["mean_aqhi"], color="#4c72b0", width=0.62,
                  edgecolor="white", label="Mean AQHI (left axis)")
    ax.set_ylabel("Mean AQHI", color="#4c72b0")
    ax.set_ylim(2.7, 4.3)
    ax.set_xlabel("Year")
    ax.set_xticks(y["year"])
    ax.set_xticklabels([str(int(v)) if v != 2026 else "2026\n(H1)"
                        for v in y["year"]], fontsize=9.5)
    ax.set_title("Fig. 4  Annual mean AQHI and high-pollution share")

    axb = ax.twinx()
    axb.plot(y["year"], y["high_pct"], color="#c0392b", marker="o", ms=5, lw=1.8,
             label="Share AQHI >= 7 (right axis)")
    axb.set_ylabel("Share of hours with AQHI >= 7 (%)", color="#c0392b")
    axb.set_ylim(0, 6.2)
    axb.grid(False)
    axb.spines["right"].set_visible(True)

    # Bar labels must be drawn on the axb layer, otherwise the line on top covers them
    def to_right(v):
        return 0.0 + (v - 2.7) / (4.3 - 2.7) * 6.2

    for b, v in zip(bars, y["mean_aqhi"]):
        axb.text(b.get_x() + b.get_width() / 2, to_right(v) + 0.19, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=8.5, color="0.15", zorder=10,
                 bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none"))

    for xv, vv in zip(y["year"], y["high_pct"]):
        if xv in (2014, 2023, 2025):
            axb.annotate(f"{vv:.1f}%", (xv, vv), xytext=(-4, 10),
                         textcoords="offset points", fontsize=8.5, color="#c0392b",
                         bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                   ec="none", alpha=0.92))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False)
    ax.text(0.01, -0.2, "2013 excluded (only 30-31 Dec). 2026 covers January-June only.",
            transform=ax.transAxes, fontsize=9, color="0.25")
    save(fig, "fig4_yearly_trend.png")
    return y


# fig5 Monthly seasonality
def fig5(m):
    mo = m.groupby("month").agg(mean_aqhi=("aqhi", "mean"),
                                high_pct=("aqhi", lambda s: (s >= 7).mean() * 100)).reset_index()
    season_of = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
                 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer",
                 9: "Autumn", 10: "Autumn", 11: "Autumn"}
    band = {"Winter": "#dce9f5", "Spring": "#e2f0dc", "Summer": "#fbe6d6", "Autumn": "#f5e6c8"}
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    # Season bands (split at calendar order to avoid a wrap-around gap)
    spans = [("Winter", 1, 2), ("Spring", 3, 5), ("Summer", 6, 8),
             ("Autumn", 9, 11), ("Winter", 12, 12)]
    for name, a, b in spans:
        ax.axvspan(a - 0.5, b + 0.5, color=band[name], zorder=0)
        ax.text((a + b) / 2, 4.47, name, ha="center", fontsize=8.5, color="0.35")

    bars = ax.bar(mo["month"], mo["mean_aqhi"], color="#4c72b0", width=0.55,
                  edgecolor="white", zorder=3)
    for b, v in zip(bars, mo["mean_aqhi"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.055, f"{v:.2f}",
                ha="center", fontsize=8.5, zorder=6,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none"))
    ax.set_ylim(2.2, 4.60)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean AQHI", color="#4c72b0")
    ax.set_title("Fig. 5  Monthly mean AQHI and high-pollution share")

    axb = ax.twinx()
    axb.plot(mo["month"], mo["high_pct"], color="#c0392b", marker="o", ms=5.5, lw=1.9,
             zorder=5)
    axb.set_ylabel("Share of hours with AQHI >= 7 (%)", color="#c0392b")
    axb.set_ylim(0, 7.4)
    axb.grid(False)
    axb.spines["right"].set_visible(True)
    top = mo.loc[mo["high_pct"].idxmax()]
    axb.annotate(f"{top['high_pct']:.1f}% in Sep", (top["month"], top["high_pct"]),
                 xytext=(-9, 13), textcoords="offset points", fontsize=8.5,
                 color="#c0392b", ha="right",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.75", alpha=0.95))

    ax.text(0.99, -0.19, "Values measured only; background bands show seasons.",
            transform=ax.transAxes, ha="right", fontsize=9, color="0.25")
    save(fig, "fig5_monthly_season.png")
    return mo


# fig6 Weekday versus weekend
def fig6(m):
    w = m.groupby(["is_weekend", "hour"])["aqhi"].mean().unstack(0)
    diff = (w[0] - w[1]).abs()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(w.index, w[0], color="#2c7fb8", lw=2, marker="o", ms=4.5,
            label="Weekday (Mon-Fri)")
    ax.plot(w.index, w[1], color="#e08214", lw=2, marker="s", ms=4, ls="--",
            label="Weekend (Sat-Sun)")
    ax.fill_between(w.index, w[0], w[1], color="0.7", alpha=0.35,
                    label="Difference")
    ax.set_xlabel("Hour of day (EPD label 01-24)")
    ax.set_ylabel("Mean AQHI")
    ax.set_title("Fig. 6  Weekday versus weekend diurnal profile")
    ax.set_xticks(range(1, 25))
    ax.set_xlim(0.6, 24.4)
    ax.legend(loc="upper left", frameon=False, ncol=3, columnspacing=1.2)
    ax.text(0.99, -0.18,
            f"Largest hourly gap: {diff.max():.2f} AQHI (hour {diff.idxmax():02d}); "
            f"average absolute gap over 24 hours: {diff.mean():.2f}",
            transform=ax.transAxes, ha="right", fontsize=9, color="0.25")
    save(fig, "fig6_weekday_weekend.png")
    return w


# fig7 Early versus recent improvement
def fig7(m):
    e = m[m["year"].isin([2014, 2015])].groupby("station")["aqhi"].mean()
    l = m[m["year"].isin([2024, 2025])].groupby("station")["aqhi"].mean()
    d = pd.DataFrame({"early": e, "late": l}).dropna()
    d["delta"] = d["late"] - d["early"]
    d = d.sort_values("delta")

    fig, ax = plt.subplots(figsize=(9.6, 7.2))
    ypos = np.arange(len(d))
    ax.hlines(ypos, d["early"], d["late"], color="0.75", lw=1.6, zorder=1)
    ax.scatter(d["early"], ypos, s=62, color="#2c7fb8", zorder=3, label="2014-2015")
    ax.scatter(d["late"], ypos, s=62, color="#c0392b", zorder=3, label="2024-2025")
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(max(r["early"], r["late"]) + 0.06, i, f"{r['delta']:+.2f}",
                va="center", fontsize=9, color="#c0392b" if r["delta"] < 0 else "0.3")
    ax.set_yticks(ypos)
    ax.set_yticklabels(d.index, fontsize=9)
    ax.set_xlabel("Mean AQHI")
    ax.set_ylabel("Station")
    ax.set_title("Fig. 7  Change in mean AQHI between 2014-2015 and 2024-2025")
    ax.legend(loc="upper right", frameon=False,
              bbox_to_anchor=(1.0, 0.28))
    lo = min(d["early"].min(), d["late"].min()) - 0.20
    hi = max(d["early"].max(), d["late"].max()) + 0.50
    ax.set_xlim(lo, hi)
    ax.text(0.01, -0.14, f"All {len(d)} stations improved. "
                         f"Range of change: {d['delta'].min():+.2f} to {d['delta'].max():+.2f}.",
            transform=ax.transAxes, fontsize=9, color="0.25")
    save(fig, "fig7_improvement.png")
    return d


# fig8 Spatial pattern
def load_hk_outline():
    """Read the Hong Kong land outline (OSM natural=coastline, ODbL). Returns [] if missing."""
    path = None
    for fn in ("hk_land.geojson", "hk_boundary.geojson"):
        p = os.path.join(BASE, "assets", fn)
        if os.path.exists(p):
            path = p
            break
    if path is None:
        return []
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    rings = []
    for feat in gj.get("features", []):
        g = feat.get("geometry") or {}
        polys = g.get("coordinates") if g.get("type") == "MultiPolygon" else [g.get("coordinates")]
        if not polys or polys[0] is None:
            continue
        for poly in polys:
            if poly and len(poly[0]) >= 3:
                rings.append(poly[0])
    return rings


def fig8(m, st):
    g = m.groupby("station")["aqhi"].mean()
    s = st.set_index("station_code")
    s["mean"] = g
    s = s.dropna(subset=["mean"]).sort_values("mean", ascending=False)
    s["idx"] = range(1, len(s) + 1)

    fig = plt.figure(figsize=(13.4, 8.2))
    ax = fig.add_axes([0.06, 0.15, 0.60, 0.74])
    ax.set_facecolor("#e8f1f7")          # Sea

    # Land basemap
    rings = load_hk_outline()
    for ring in rings:
        ax.fill([p[0] for p in ring], [p[1] for p in ring],
                facecolor="#f3f0e8", edgecolor="#9c9c9c", lw=0.7,
                zorder=1, closed=True)

    def bub(v):
        """Bubble area increases monotonically with mean AQHI; the floor keeps two-digit labels readable."""
        return 200 + (v - 3.10) / 0.80 * 400

    for t in TYPE_ORDER:
        sub = s[s["station_type"] == t]
        ax.scatter(sub["longitude"], sub["latitude"],
                   s=[bub(v) for v in sub["mean"]], color=TYPE_COLOR[t],
                   alpha=0.88, edgecolor="white", lw=1.2, label=t, zorder=3)
    # Station names overlap at this scale; use in-bubble index numbers plus the side table
    for _, row in s.iterrows():
        ax.text(row["longitude"], row["latitude"], str(row["idx"]),
                ha="center", va="center", fontsize=7.2, color="white",
                fontweight="bold", zorder=5)
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.set_title("Fig. 8  Spatial pattern of mean AQHI (bubble area increases with mean AQHI)")
    ax.set_xlim(113.80, 114.52)
    ax.set_ylim(22.13, 22.58)
    ax.set_aspect(1 / np.cos(np.deg2rad(22.35)))
    ax.legend(title="Station type", loc="lower right", frameon=False)

    # Size legend so the bubble sizes map to values
    x0, y0 = 0.05, 0.075
    for i, v in enumerate([3.2, 3.5, 3.8]):
        yy = y0 + i * 0.065
        ax.scatter([x0], [yy], s=bub(v), color="0.55", alpha=0.9,
                   edgecolor="white", lw=1.0, transform=ax.transAxes,
                   zorder=6, clip_on=False)
        ax.text(x0 + 0.048, yy, f"{v:.1f}", fontsize=8.2, va="center",
                transform=ax.transAxes, color="0.15", zorder=6)
    ax.text(x0 - 0.008, y0 + 3 * 0.065 + 0.004, "Mean AQHI", fontsize=8.5,
            fontweight="bold", transform=ax.transAxes, color="0.15", zorder=6,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.8))

    # Side station lookup table
    axt = fig.add_axes([0.705, 0.15, 0.275, 0.74])
    axt.set_axis_off()
    axt.set_xlim(0, 1)
    axt.set_ylim(0, 1)
    cols = {"no": 0.0, "name": 0.09, "mean": 0.93}
    axt.text(cols["no"], 1.03, "No", fontsize=9, fontweight="bold")
    axt.text(cols["name"], 1.03, "Station", fontsize=9, fontweight="bold")
    axt.text(cols["mean"], 1.03, "Mean", fontsize=9, fontweight="bold", ha="right")
    axt.plot([0, 0.93], [0.995, 0.995], color="0.6", lw=0.8,
             clip_on=False, scalex=False, scaley=False)
    for i, (name, row) in enumerate(s.iterrows()):
        yy = 0.945 - i * 0.0525
        axt.text(cols["no"], yy, f"{row['idx']}", fontsize=8.4, color="0.25")
        axt.text(cols["name"], yy, name, fontsize=8.4)
        axt.text(cols["mean"], yy, f"{row['mean']:.2f}", fontsize=8.4, ha="right")

    fig.text(0.06, 0.055,
             "Bubble area increases with mean AQHI (see size key); "
             "colours follow station type. Measured values only.",
             fontsize=9, color="0.25")
    fig.text(0.06, 0.022,
             "Basemap: coastline (OpenStreetMap contributors, ODbL); "
             "land beyond the HK boundary is shown for context.",
             fontsize=8, color="0.45")
    save(fig, "fig8_station_map.png")
    return s


def write_summary(m):
    """Export the numbers behind the eight figures as text for direct reuse in the report and slides."""
    L = []
    add = L.append
    add("FIGURE DATA SUMMARY")
    add("=" * 70)
    add("Source: output/aqhi_long.csv, measured values only (estimated and")
    add(f"missing rows excluded). Records used: {len(m):,}")
    add("")

    add("FIG 1 - Diurnal profile (all 18 stations)")
    h = m.groupby("hour")["aqhi"].mean()
    add("  hour   mean AQHI")
    for k, v in h.items():
        add(f"  {k:02d}     {v:.3f}")
    add(f"  Lowest  : {h.min():.3f} at hour {h.idxmin():02d}")
    add(f"  Highest : {h.max():.3f} at hour {h.idxmax():02d}")
    add(f"  Amplitude: {h.max()-h.min():.3f} ({(h.max()-h.min())/h.mean()*100:.1f}% of the mean)")
    add("  Same hours, three representative stations (mean AQHI):")
    for s in ["Causeway Bay", "Sha Tin", "Tap Mun"]:
        v = m[m["station"] == s].groupby("hour")["aqhi"].mean()
        add(f"    {s:<14s} lowest {v.min():.3f} (hour {v.idxmin():02d})  "
            f"highest {v.max():.3f} (hour {v.idxmax():02d})")
    add("")

    add("FIG 2 - Station x hour heatmap")
    mat = m.pivot_table(index="station", columns="hour", values="aqhi", aggfunc="mean")
    add(f"  Highest cell: {mat.values.max():.3f} "
        f"({mat.stack().idxmax()[0]}, hour {mat.stack().idxmax()[1]:02d})")
    add(f"  Lowest cell : {mat.values.min():.3f} "
        f"({mat.stack().idxmin()[0]}, hour {mat.stack().idxmin()[1]:02d})")
    add("")

    add("FIG 3 - Station ranking (mean AQHI and share of hours AQHI >= 7)")
    g = m.groupby("station").agg(mean=("aqhi", "mean"),
                                 high=("aqhi", lambda s: (s >= 7).mean() * 100),
                                 n=("aqhi", "size")).sort_values("mean", ascending=False)
    add("  station            mean   share>=7   records")
    for k, r in g.iterrows():
        add(f"  {k:<18s} {r['mean']:.3f}   {r['high']:5.2f}%   {int(r['n']):,}")
    add("")

    add("FIG 4 - Annual trend (2014 onward; 2026 = Jan-Jun)")
    y = m[m["year"] >= 2014].groupby("year").agg(
        mean=("aqhi", "mean"), high=("aqhi", lambda s: (s >= 7).mean() * 100))
    add("  year    mean    share>=7")
    for k, r in y.iterrows():
        add(f"  {int(k)}   {r['mean']:.3f}   {r['high']:5.2f}%")
    add("")

    add("FIG 5 - Monthly pattern")
    mo = m.groupby("month").agg(mean=("aqhi", "mean"),
                                high=("aqhi", lambda s: (s >= 7).mean() * 100))
    add("  month   mean    share>=7")
    for k, r in mo.iterrows():
        add(f"  {int(k):02d}     {r['mean']:.3f}   {r['high']:5.2f}%")
    add(f"  Highest mean month: {mo['mean'].idxmax()} ({mo['mean'].max():.3f}); "
        f"lowest: {mo['mean'].idxmin()} ({mo['mean'].min():.3f})")
    add(f"  Highest >=7 share : {mo['high'].idxmax()} ({mo['high'].max():.2f}%); "
        f"lowest: {mo['high'].idxmin()} ({mo['high'].min():.2f}%)")
    add("")

    add("FIG 6 - Weekday versus weekend")
    w = m.groupby(["is_weekend", "hour"])["aqhi"].mean().unstack(0)
    d = (w[0] - w[1]).abs()
    add(f"  Largest gap: {d.max():.3f} at hour {d.idxmax():02d}")
    add(f"  Average absolute gap over 24 hours: {d.mean():.3f}")
    add("")

    add("FIG 7 - Change between 2014-2015 and 2024-2025")
    e = m[m["year"].isin([2014, 2015])].groupby("station")["aqhi"].mean()
    l = m[m["year"].isin([2024, 2025])].groupby("station")["aqhi"].mean()
    dd = pd.DataFrame({"early": e, "late": l}).dropna()
    dd["delta"] = dd["late"] - dd["early"]
    dd = dd.sort_values("delta")
    add("  station            2014-15  2024-25   change")
    for k, r in dd.iterrows():
        add(f"  {k:<18s} {r['early']:.3f}    {r['late']:.3f}    {r['delta']:+.3f}")
    add(f"  Stations compared: {len(dd)}; improved: {(dd['delta'] < 0).sum()}")
    add("")

    add("FIG 8 - Spatial pattern (mean AQHI by station, with coordinates)")
    st = pd.read_csv(os.path.join(OUT, "stations.csv"))
    s8 = st.set_index("station_code")
    s8["mean"] = m.groupby("station")["aqhi"].mean()
    s8 = s8.dropna(subset=["mean"]).sort_values("mean", ascending=False)
    add("  No  station            type       lat        lon        mean")
    for i, (k, r) in enumerate(s8.iterrows(), start=1):
        add(f"  {i:2d}  {k:<18s} {r['station_type']:<10s} "
            f"{r['latitude']:.5f}  {r['longitude']:.5f}  {r['mean']:.3f}")
    add("")

    add("AQHI BAND DISTRIBUTION (all measured values)")
    tot = len(m)
    bands = [("Low (1-3)", m["aqhi"] <= 3), ("Moderate (4-6)", (m["aqhi"] >= 4) & (m["aqhi"] <= 6)),
             ("High (7)", m["aqhi"] == 7), ("Very high (8-10)", m["aqhi"] >= 8)]
    for name, mask in bands:
        add(f"  {name:<16s} {mask.sum():>10,}  {mask.sum()/tot*100:5.2f}%")

    p = os.path.join(OUT, "figure_data_summary.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("  saved figure_data_summary.txt")


def main():
    print("Reading long table and station dimension ...")
    m, st = load()
    print(f"  {len(m):,} measured records (estimated and missing excluded)\n")
    print("Plotting:")
    fig1(m); fig2(m); fig3(m); fig4(m); fig5(m); fig6(m); fig7(m); fig8(m, st)
    print("\nSummary:")
    write_summary(m)
    print("\nDone. Outputs: figures/ and output/figure_data_summary.txt")


if __name__ == "__main__":
    main()
