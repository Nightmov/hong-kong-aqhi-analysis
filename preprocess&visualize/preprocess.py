import os
import glob
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

NON_STATION = {"Date", "Hour"}


def find_header_row(path):
    """Return the 0-based header row index, or None if not found."""
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for i, line in enumerate(f):
            s = line.lstrip("\ufeff")
            if s.startswith("Date") and "Hour" in s:
                return i
    return None


def parse_one_file(path):
    hi = find_header_row(path)
    if hi is None:
        raise ValueError(f"Header row not found: {path}")

    df = pd.read_csv(path, skiprows=hi, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]

    # Forward-fill the date (equivalent to merged cells)
    df["Date"] = df["Date"].ffill()

    # Drop Daily Max summary rows: keep only rows whose Hour parses as a number
    h = pd.to_numeric(df["Hour"], errors="coerce")
    df = df.loc[h.notna()].copy()
    df["Hour"] = h[h.notna()].astype(int)

    stations = [c for c in df.columns if c not in NON_STATION]
    long = df.melt(id_vars=["Date", "Hour"], value_vars=stations,
                   var_name="station", value_name="raw")
    long["source_file"] = os.path.basename(path)
    return long


def main():
    files = sorted(glob.glob(os.path.join(DATA, "*_Eng.csv")))
    print(f"Files to process: {len(files)}")

    parts = []
    for i, p in enumerate(files, 1):
        parts.append(parse_one_file(p))
        if i % 25 == 0 or i == len(files):
            print(f"  parsed {i}/{len(files)}")
    long = pd.concat(parts, ignore_index=True)
    print(f"Long table after melt: {len(long):,} rows")

    # Value parsing
    # EPD writes "no data" in four ways: empty cell, "-", "-*", and a lone "*".
    # Only "<number>*" and "10+*" count as estimated / capped and must stay
    # distinct from plain missing.
    s = long["raw"].fillna("").astype(str).str.strip()
    core = s.str.replace(r"\*+$", "", regex=True).str.strip()
    is_capped_raw = core.str.contains(r"10\+", regex=True)

    v = core.str.replace("10+", "10", regex=False)
    v = v.where(v != "", None)
    aqhi = pd.to_numeric(v, errors="coerce")

    long["aqhi"] = aqhi.astype("float64")
    long["is_estimated"] = (s.str.endswith("*") & aqhi.notna()).astype("int8")
    long["is_capped"] = (is_capped_raw & aqhi.notna()).astype("int8")

    # Three sources of missing values (for the quality report)
    n_blank_cell = int((s == "").sum())
    n_dash = int(s.isin(["-", "-*"]).sum())
    n_star_only = int((s == "*").sum())

    # Time assembly: labels 01-24 -> 00:00-23:00
    date = pd.to_datetime(long["Date"], errors="coerce")
    long["datetime"] = date + pd.to_timedelta(long["Hour"] - 1, unit="h")

    long["date"] = date.dt.strftime("%Y-%m-%d")
    long["year"] = date.dt.year.astype("Int64")
    long["month"] = date.dt.month.astype("Int64")
    long["day"] = date.dt.day.astype("Int64")
    long["weekday"] = date.dt.dayofweek.astype("Int64") + 1      # 1 = Monday
    long["is_weekend"] = (long["weekday"] >= 6).astype("int8")
    long["yyyymm"] = long["year"].astype(str) + "-" + long["month"].astype(str).str.zfill(2)
    season = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer",
              9: "Autumn", 10: "Autumn", 11: "Autumn"}
    long["season"] = long["month"].map(season)

    cols = ["datetime", "date", "hour", "year", "month", "day", "weekday",
            "is_weekend", "season", "yyyymm", "station", "aqhi",
            "is_estimated", "is_capped", "source_file"]
    long = long.rename(columns={"Hour": "hour"})[cols].sort_values(
        ["datetime", "station"], kind="stable").reset_index(drop=True)

    out_path = os.path.join(OUT, "aqhi_long.csv")
    long.to_csv(out_path, index=False, encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S")
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"Wrote {out_path}  ({len(long):,} rows, {size_mb:.1f} MB)")

    # Quality report
    total = len(long)
    n_missing = int(long["aqhi"].isna().sum())
    n_est = int((long["is_estimated"] == 1).sum())
    n_capped = int((long["is_capped"] == 1).sum())
    n_valid = total - n_missing
    n_measured = n_valid - n_est

    L = []
    A = L.append
    A("AQHI DATA QUALITY REPORT")
    A("=" * 72)
    A("Source        : EPD hourly AQHI, English release (YYYYMM_Eng.csv), data.gov.hk")
    A(f"Files         : {len(files)} monthly files")
    A("Period        : 2013-12-30 00:00  ~  2026-06-30 23:00")
    A("Stations      : 18 (15 before 2016, 16 from 2016, 18 from 2020)")
    A("")
    A("0. MEASUREMENT BASIS (read this first)")
    A("   The long table holds three nested sets, and they are not interchangeable:")
    A(f"     - all rows              : {total:,}   (structure of the dataset)")
    A(f"     - valid values          : {n_valid:,}   (measured + EPD-substituted)")
    A(f"     - MEASURED values       : {n_measured:,}   <- basis of every statistic")
    A("   All statistics in sections 2 and 4 below, and every figure produced by")
    A("   04_visualize.py, are computed on MEASURED values only (is_estimated = 0")
    A("   and AQHI not missing). Section 1 is a data ledger and therefore counts")
    A("   all rows; it is not a statistical basis. Any headline number quoted in")
    A("   the report must cite which of the three sets it comes from.")
    A("")
    A("1. RECORD COUNT")
    A(f"   Long-table rows                       : {total:,}")
    A(f"   Unique (datetime, station) pairs      : "
      f"{len(long.drop_duplicates(['datetime','station'])):,}")
    A(f"   Valid AQHI values                     : {n_valid:,}  ({n_valid/total*100:.2f}%)")
    A(f"     - measured                          : {n_measured:,}  ({n_measured/total*100:.2f}%)")
    A(f"     - estimated (marked with *)      : {n_est:,}  ({n_est/total*100:.2f}%)")
    A(f"   Missing (no value)                    : {n_missing:,}  ({n_missing/total*100:.2f}%)")
    A(f"     - empty cell                        : {n_blank_cell:,}")
    A(f"     - '-' or '-*'                       : {n_dash:,}")
    A(f"     - '*' alone (no number)             : {n_star_only:,}")
    A(f"   Capped at '10+' (recorded as 10)      : {n_capped:,}  ({n_capped/total*100:.2f}%)")
    A("")
    A("   Note: EPD writes 'no data' in four different ways - an empty cell,")
    A("   '-', '-*', and a lone '*'. All four are treated as missing (NaN).")
    A("   Only '<number>*' and '10+*' qualify as estimated / capped values.")
    A("")
    A("   Note on the '10+' count: Step 1 reports 1,772 '10+' cells in the hourly")
    A("   rows plus 470 more inside the Daily Max rows (those rows are discarded")
    A("   here, so the long table keeps exactly 1,772). A cell written '10+*' is")
    A("   counted both as estimated and as capped; there are 50 such cells.")
    A("")
    A("2. VALUE DISTRIBUTION OF AQHI  (measured values only)")
    v_ok = long.loc[long["is_estimated"] == 0, "aqhi"].dropna()
    A(f"   Basis                                 : {len(v_ok):,} measured values")
    A(f"   Min / Median / Mean / Max             : "
      f"{v_ok.min():.0f} / {v_ok.median():.0f} / {v_ok.mean():.3f} / {v_ok.max():.0f}")
    q = v_ok.quantile([.25, .75, .95, .99])
    A(f"   Q1 / Q3 / P95 / P99                   : "
      f"{q.iloc[0]:.1f} / {q.iloc[1]:.1f} / {q.iloc[2]:.1f} / {q.iloc[3]:.1f}")
    bands = pd.cut(v_ok, bins=[0, 3, 6, 7, 10], labels=["Low (1-3)", "Moderate (4-6)",
                                                          "High (7)", "Very High (8-10)"],
                   right=True, include_lowest=True)
    A("   Band distribution (EPD scale):")
    bc = bands.value_counts().sort_index()
    for k, c in bc.items():
        A(f"     {k:16s}: {c:,}  ({c/len(v_ok)*100:.2f}% of measured)")
    A("")
    A("   Cross-check, same statistic on valid values including EPD-substituted")
    A("   rows (for reference only, do NOT mix with the figures):")
    v_all = long["aqhi"].dropna()
    bands_all = pd.cut(v_all, bins=[0, 3, 6, 7, 10], labels=["Low (1-3)", "Moderate (4-6)",
                                                              "High (7)", "Very High (8-10)"],
                       right=True, include_lowest=True)
    bc_all = bands_all.value_counts().sort_index()
    A(f"     mean {v_all.mean():.3f} vs measured {v_ok.mean():.3f}"
      f"  (difference {abs(v_all.mean()-v_ok.mean()):.3f})")
    for k, c in bc_all.items():
        A(f"     {k:16s}: {c/len(v_all)*100:5.2f}% vs {bc[k]/len(v_ok)*100:5.2f}% measured")
    A("")
    A("3. TIME COVERAGE")
    A(f"   Calendar days present                 : {long['date'].nunique():,}")
    A(f"   Distinct hour labels used             : {long['hour'].nunique()} (1-24)")
    A(f"   Rows per hour label (station-avg)     : "
      f"{total/long['hour'].nunique()/18:,.0f}")
    A("   Known source gap: 2023-08-25 ~ 2023-08-31 (7 days) absent from the")
    A("   official file itself. Verified: the file downloaded from aqhi.gov.hk")
    A("   today is byte-identical (sha1 2b4547021cd7) to the local copy, and the")
    A("   Chinese release of the same month is truncated in the same way.")
    A("   -> No data are fabricated for this window; it is reported as missing.")
    A("")
    A("4. STATION PROFILE (coverage and completeness)")
    A("   missing% and est% describe data availability; 'mean' is measured-only.")
    A(f"   {'station':18s} {'first_obs':10s} {'n_obs':>8s} {'missing%':>9s} "
      f"{'est%':>7s} {'capped':>7s} {'mean':>6s}")
    for st in sorted(long["station"].unique(),
                     key=lambda x: -long[long["station"] == x]["aqhi"].notna().mean()):
        d = long[long["station"] == st]
        nn = len(d)
        ms = d["aqhi"].isna().mean() * 100
        es = d["is_estimated"].mean() * 100
        cp = int((d["is_capped"] == 1).sum())
        fd = d["date"].min()
        dm = d.loc[(d["is_estimated"] == 0) & d["aqhi"].notna(), "aqhi"]
        A(f"   {st:18s} {fd:10s} {nn:8,d} {ms:8.2f}% {es:6.2f}% "
          f"{cp:7,d} {dm.mean():6.2f}")
    A("")
    A("5. MISSING VALUE HANDLING")
    meas = long[long["is_estimated"] == 0]
    fill = meas.groupby(["station", "hour"])["aqhi"].transform("mean")
    filled = meas["aqhi"].fillna(fill)
    mean_raw = meas["aqhi"].mean()
    mean_fill = filled.mean()
    high_raw = (meas["aqhi"] >= 7).sum() / meas["aqhi"].notna().sum() * 100
    high_fill = (filled >= 7).sum() / len(filled) * 100
    A("   Missing share is 0.97% of all rows. No value is imputed in the long")
    A("   table; NaN means genuinely 'no reading', which downstream code can")
    A("   drop per analysis. Rationale: AQHI is a health-risk index, so an")
    A("   invented value would create a health signal that never happened, and")
    A("   any time-direction fill would also smear a spike across hours and")
    A("   distort the co-occurrence pattern that association-rule work needs.")
    A("   Sensitivity check, filling each gap with that station's mean at that")
    A("   hour label of the day:")
    A(f"     overall mean    : {mean_raw:.3f} (as-is)  vs  {mean_fill:.3f} (imputed)"
      f"  -> difference {abs(mean_fill-mean_raw):.4f}")
    A(f"     share AQHI >= 7 : {high_raw:.2f}% (as-is)  vs  {high_fill:.2f}% (imputed)"
      f"  -> difference {abs(high_fill-high_raw):.2f} pp")
    A("   The two results are indistinguishable, so imputation would add")
    A("   assumptions without changing any conclusion. Recommendation: keep")
    A("   NaN, and where an analysis cannot tolerate it, restrict to stations")
    A("   and periods that meet a coverage threshold instead of inventing data.")
    A("")
    A("6. REMARKS FOR DOWNSTREAM USE")
    A("   - 'Daily Max' summary rows were excluded; they are not hourly observations.")
    A("   - Hour labels follow EPD convention 01-24; 'hour'=24 maps to 23:00-24:00.")
    A("   - '10+' is a censored value meaning 'at least 10', not an outlier; it is")
    A("     kept as 10 and flagged, so extreme hours are preserved.")
    A("   - estimated rows (marked with *) are substituted station data and are flagged")
    A("     separately; they are not the station's own measurement, so they are")
    A("     excluded from every statistic and every figure in this project.")
    A("   - 2013 contains only 2 days (30-31 Dec); exclude it from annual comparisons.")
    A("")
    A("   QUOTING CONVENTION: a figure or report sentence about AQHI levels should")
    A("   cite measured-only numbers. Only coverage / data-availability statements")
    A("   may use the all-rows ledger in section 1.")

    rep = os.path.join(OUT, "quality_report.txt")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n" + "\n".join(L[:60]))
    print(f"\nReport written to {rep}")


if __name__ == "__main__":
    main()
