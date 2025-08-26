# ♿ Difficulty climbing stairs — Household projections dashboard
# Data: data/prep_PEC19.csv, data/raw_F4006.csv, data/Households_size_2022.csv
# Method: Q15(c) prevalence (C12 + C13) × demography → person share → household share;
#         plus buggy proxy = all children aged 0–3 (not filtered by disability).

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Households with difficulty climbing stairs — Projections",
    page_icon="♿",
    layout="wide"
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
YEARS_FOCUS = [2022, 2030, 2040, 2050]

# -------------- Helpers ---------------------
def band_from_age(age: pd.Series) -> pd.Series:
    """Map single ages to bands '0-3', '4-64', '65+'."""
    out = pd.Series(index=age.index, dtype="object")
    out[age <= 3] = "0-3"
    out[(age >= 4) & (age <= 64)] = "4-64"
    out[age >= 65] = "65+"
    return out

def person_share_to_household_share(q: pd.Series, hh_df: pd.DataFrame) -> pd.Series:
    """
    Convert person share q to household share p using size model:
    p = Σ_s h_s * [1 - (1 - q)^s], treating '7+' as size 7.
    Vectorized over q (Series indexed by year).
    """
    hh = hh_df.copy()
    hh["size_numeric"] = pd.to_numeric(hh["size_numeric"], errors="coerce").fillna(7).astype(int)
    hs = (hh["households_2022"] / hh["households_2022"].sum()).to_numpy()  # weight by size class
    s = hh["size_numeric"].to_numpy()                                       # sizes 1..7
    q_vals = q.to_numpy()[:, None]                                          # (n_years, 1)
    # probability ≥1 affected in a size-s household, averaged over size distribution
    p_vals = (1.0 - (1.0 - q_vals) ** s) * hs                               # (n_years, n_sizes)
    return pd.Series(p_vals.sum(axis=1), index=q.index)

def to_title(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.title()

# -------------- Data load & prep ------------
@st.cache_data(show_spinner=True)
def load_inputs():
    # ---------- Population (PEC19) ----------
    pop = pd.read_csv(DATA_DIR / "prep_PEC19.csv")
    pop.columns = [c.strip() for c in pop.columns]

    # Filter to Method - M2 if column exists
    if "Criteria for Projection" in pop.columns:
        m = pop["Criteria for Projection"].astype(str)
        pop = pop[m.str.contains("Method", na=False) & m.str.contains("M2", na=False)]

    # Standardize population column name to VALUE
    if "VALUE" not in pop.columns:
        for cand in ("Value", "value", "Population", "population"):
            if cand in pop.columns:
                pop.rename(columns={cand: "VALUE"}, inplace=True)
                break
    if "VALUE" not in pop.columns:
        raise ValueError("prep_PEC19.csv must have 'VALUE' (or 'Population').")
    pop["VALUE"] = pd.to_numeric(pop["VALUE"], errors="coerce").fillna(0)

    # Ensure Year/Sex/Band
    if "Band" not in pop.columns:
        if "AgeNum" in pop.columns:
            pop["Band"] = band_from_age(pd.to_numeric(pop["AgeNum"], errors="coerce"))
        else:
            raise ValueError("prep_PEC19.csv needs 'Band' or 'AgeNum'.")
    pop["Sex"] = to_title(pop["Sex"])
    pop = pop[pop["Sex"].isin(["Male", "Female"])]

    # Focus years
    years_present = sorted(pop["Year"].unique().tolist())
    years = [y for y in YEARS_FOCUS if y in years_present] or years_present
    pop = pop[pop["Year"].isin(years)]

    # Aggregate to Year × Sex × Band
    pop_g = (
        pop.groupby(["Year", "Sex", "Band"], as_index=False)["VALUE"]
           .sum()
           .rename(columns={"VALUE": "ProjectedPopulation"})
    )

    # ---------- Baseline rates from raw_F4006 (Q15c = C12 + C13 only) ----------
    raw = pd.read_csv(DATA_DIR / "raw_F4006.csv")
    raw.columns = [c.strip() for c in raw.columns]

    # Normalize value column
    if "VALUE" not in raw.columns:
        for cand in ("Value", "value", "Count", "count"):
            if cand in raw.columns:
                raw.rename(columns={cand: "VALUE"}, inplace=True)
                break
    if "VALUE" not in raw.columns:
        raise ValueError("raw_F4006.csv must have a numeric 'VALUE' column.")

    # If Year exists, keep 2022 snapshot
    if "Year" in raw.columns:
        raw = raw[raw["Year"] == 2022]

    # Q15(c) “to some extent” (C12) + “to a great extent” (C13)
    keep_codes = {"F4006C12", "F4006C13"}
    if "STATISTIC" not in raw.columns:
        raise ValueError("raw_F4006.csv must include a 'STATISTIC' column (e.g., F4006C12, F4006C13).")
    raw_q15c = raw[raw["STATISTIC"].astype(str).isin(keep_codes)].copy()

    # Require Band & Sex
    if "Band" not in raw_q15c.columns:
        raise ValueError("raw_F4006.csv must include 'Band' (0-3, 4-64, 65+).")
    if "Sex" not in raw_q15c.columns:
        raise ValueError("raw_F4006.csv must include 'Sex'.")
    raw_q15c["Sex"] = to_title(raw_q15c["Sex"])

    # Numerics
    raw_q15c["VALUE"] = pd.to_numeric(raw_q15c["VALUE"], errors="coerce").fillna(0)

    # Q15c counts by Band×Sex (2022)
    stairs_2022 = (
        raw_q15c.groupby(["Band", "Sex"], as_index=False)["VALUE"]
                .sum()
                .rename(columns={"VALUE": "Q15cCount2022"})
    )

    # Denominator: total 2022 population by Band×Sex from PEC19
    pop_2022 = (
        pop_g[pop_g["Year"] == 2022]
        .groupby(["Band", "Sex"], as_index=False)["ProjectedPopulation"]
        .sum()
        .rename(columns={"ProjectedPopulation": "TotalPop2022"})
    )

    base = stairs_2022.merge(pop_2022, on=["Band", "Sex"], how="left")
    if base["TotalPop2022"].isna().any():
        missing = base[base["TotalPop2022"].isna()][["Band", "Sex"]].drop_duplicates()
        raise ValueError(f"Missing 2022 population for Band×Sex combos in PEC19: \n{missing}")

    base["baseline_rate"] = (base["Q15cCount2022"] / base["TotalPop2022"]).clip(0, 1)

    # ---------- Apply baseline to all years ----------
    proj = pop_g.merge(base[["Band", "Sex", "baseline_rate"]], on=["Band", "Sex"], how="left")
    if proj["baseline_rate"].isna().any():
        missing = proj[proj["baseline_rate"].isna()][["Band", "Sex"]].drop_duplicates()
        raise ValueError(f"Missing baseline_rate for Band×Sex combos: \n{missing}")

    # Expected persons with Q15c difficulty (incl. climbing stairs)
    proj["ProjectedStairDifficulty"] = proj["ProjectedPopulation"] * proj["baseline_rate"]

    # BUGGY proxy: all children aged 0–3 (regardless of disability)
    proj["Child0_3_Buggy"] = np.where(proj["Band"] == "0-3", proj["ProjectedPopulation"], 0.0)

    # ---------- Household size mix (2022) ----------
    hh = pd.read_csv(DATA_DIR / "Households_size_2022.csv")
    hh.columns = [c.strip() for c in hh.columns]
    needed = {"size_numeric", "households_2022"}
    if not needed.issubset(set(hh.columns)):
        raise ValueError("Households_size_2022.csv needs columns: size_numeric, households_2022")
    hh["size_numeric"] = pd.to_numeric(hh["size_numeric"], errors="coerce").fillna(7).astype(int)
    hh["households_2022"] = pd.to_numeric(hh["households_2022"], errors="coerce").fillna(0)

    return proj, hh, years, base

@st.cache_data(show_spinner=True)
def build_views(proj: pd.DataFrame, hh: pd.DataFrame):
    # Totals & shares by year
    by_year = (
        proj.groupby("Year", as_index=False)
            .agg(total_pop=("ProjectedPopulation", "sum"),
                 stairs=("ProjectedStairDifficulty", "sum"),
                 child03_buggy=("Child0_3_Buggy", "sum"))
    )
    # Person shares
    by_year["q_member_difficulty"] = by_year["stairs"] / by_year["total_pop"]   # for households with member having Q15c
    by_year["q_buggy_child"]       = by_year["child03_buggy"] / by_year["total_pop"]  # for households with child 0–3

    # Household percentages
    by_year["p_member"] = person_share_to_household_share(by_year["q_member_difficulty"], hh)
    by_year["p_buggy"]  = person_share_to_household_share(by_year["q_buggy_child"], hh)
    by_year["p_either"] = by_year["p_member"] + by_year["p_buggy"] - by_year["p_member"] * by_year["p_buggy"]

    # Labels for charts
    band_long = {"0-3": "Ages 0–3 years", "4-64": "Ages 4–64 years", "65+": "Ages 65 years and over"}
    proj_lab = proj.copy()
    proj_lab["BandLabel"] = proj_lab["Band"].map(band_long)

    return by_year, proj_lab

# -------------- Load everything -------------
proj, hh, years, base_rates = load_inputs()
by_year, proj_labeled = build_views(proj, hh)

# ---------------- UI -----------------------
st.title("♿ Households affected by difficulty climbing stairs — Projections")
st.caption("Member difficulty = Census Q15(c) “to some extent” + “to a great extent” (F4006C12 + F4006C13).  Buggy = any child aged 0–3.  Projections use CSO PEC19 (Method M2).")

# Sidebar filters
st.sidebar.header("Filters")
year_opt = ["All Years"] + [str(y) for y in years]
sel_year = st.sidebar.selectbox("Select Year", options=year_opt, index=0)
sex_opt  = sorted(proj_labeled["Sex"].unique().tolist())
band_opt = ["0-3", "4-64", "65+"]
sel_sex  = st.sidebar.multiselect("Select Sex", options=sex_opt, default=sex_opt)
sel_band = st.sidebar.multiselect("Select Age Band", options=band_opt, default=band_opt)

if not sel_sex or not sel_band:
    st.warning("Please select at least one sex and one age band.")
    st.stop()

# -------------- Top KPIs (2022 snapshot) -------------------
col1, col2, col3, col4 = st.columns(4)
yr2022 = 2022 if 2022 in years else years[0]
snap = proj_labeled[proj_labeled["Year"] == yr2022]
col1.metric(f"Total population ({yr2022})", f"{int(snap['ProjectedPopulation'].sum()):,}")
col2.metric("People with difficulty (Q15c)", f"{int(snap['ProjectedStairDifficulty'].sum()):,}")
col3.metric("Children aged 0–3 (buggy)", f"{int(snap['Child0_3_Buggy'].sum()):,}")
h_row = by_year[by_year["Year"] == yr2022].iloc[0]
col4.metric("Households affected (either: member difficulty OR buggy)", f"{100*h_row['p_either']:.2f}%")

st.markdown("---")

# -------------- All-years or single-year ---
if sel_year == "All Years":
    # Persons with Q15c — grouped+stacked by Year→Sex; stacks=age bands
    st.subheader("Individuals with difficulty (Q15c) — clustered by year and sex; stacks = age bands")
    df_viz = proj_labeled[(proj_labeled["Sex"].isin(sel_sex)) &
                          (proj_labeled["Band"].isin(sel_band))].copy()
    df_viz["Year_str"] = df_viz["Year"].astype(str)
    cat_orders = {
        "Year_str": [str(y) for y in years],
        "Sex": ["Male", "Female"],
        "Band": band_opt,
        "BandLabel": ["Ages 0–3 years", "Ages 4–64 years", "Ages 65 years and over"]
    }
    fig1 = px.bar(
        df_viz,
        x=["Year_str", "Sex"], y="ProjectedStairDifficulty",
        color="BandLabel", barmode="stack", template="plotly_white",
        category_orders=cat_orders,
        labels={"Year_str": "Year", "Sex": "Sex",
                "ProjectedStairDifficulty": "People (count)", "BandLabel": "Age band"},
        title="Year → Sex clusters; stacks = age bands"
    )
    fig1.update_layout(bargap=0.25, legend_title="Age band")
    fig1.update_yaxes(tickformat=",")
    st.plotly_chart(fig1, use_container_width=True)

    # Children 0–3 (buggy) — counts by year × sex
    st.subheader("Children aged 0–3 (buggy) — people by year and sex")
    child_viz = (proj_labeled.groupby(["Year", "Sex"], as_index=False)["Child0_3_Buggy"].sum())
    child_viz = child_viz[child_viz["Sex"].isin(sel_sex)].copy()
    child_viz["Year_str"] = child_viz["Year"].astype(str)
    fig_child = px.bar(
        child_viz,
        x=["Year_str", "Sex"], y="Child0_3_Buggy",
        barmode="group", template="plotly_white",
        category_orders={"Year_str": [str(y) for y in years], "Sex": ["Male", "Female"]},
        labels={"Year_str": "Year", "Sex": "Sex", "Child0_3_Buggy": "People (count)"},
        title="Year → Sex clusters (children 0–3, buggy)"
    )
    fig_child.update_yaxes(tickformat=",")
    st.plotly_chart(fig_child, use_container_width=True)

    # Household shares — three lines: member, buggy, either
    st.subheader("Estimated share of households affected (percentages)")
    fig2 = px.line(
        by_year, x="Year", y=["p_member", "p_buggy", "p_either"],
        markers=True, template="plotly_white",
        labels={"value": "Share of households", "variable": "Measure"},
        title="Households with: ≥1 member with difficulty (Q15c), ≥1 child aged 0–3 (buggy), or either"
    )
    # Convert to %
    for tr in fig2.data:
        tr.y = [100*v for v in tr.y]
    fig2.for_each_trace(lambda t: t.update(name={
        "p_member": "Member with difficulty (Q15c)",
        "p_buggy":  "Child aged 0–3 (buggy)",
        "p_either": "Either: member difficulty OR buggy"
    }[t.name]))
    fig2.update_yaxes(title="Share of households (%)", rangemode="tozero")
    st.plotly_chart(fig2, use_container_width=True)

else:
    year_int = int(sel_year)
    st.subheader(f"Detailed projections for {year_int}")

    df_y = proj_labeled[(proj_labeled["Year"] == year_int) &
                        (proj_labeled["Sex"].isin(sel_sex)) &
                        (proj_labeled["Band"].isin(sel_band))].copy()

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("People with difficulty (Q15c)", f"{int(df_y['ProjectedStairDifficulty'].sum()):,}")
    c2.metric("Children aged 0–3 (buggy)", f"{int(df_y['Child0_3_Buggy'].sum()):,}")
    h_row = by_year[by_year["Year"] == year_int].iloc[0]
    c3.metric("Households affected (either: member difficulty OR buggy)", f"{100*h_row['p_either']:.2f}%")

    # Persons with Q15c — grouped by age band × sex
    fig_bar = px.bar(
        df_y, x="Band", y="ProjectedStairDifficulty", color="Sex",
        barmode="group", template="plotly_white",
        category_orders={"Band": ["0-3", "4-64", "65+"], "Sex": ["Male", "Female"]},
        labels={"Band": "Age band", "ProjectedStairDifficulty": "People (count)"},
        title=f"Individuals with difficulty (Q15c), by age band and sex — {year_int}"
    )
    fig_bar.update_yaxes(tickformat=",")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Children 0–3 (buggy) by sex for the selected year
    child_y = (proj_labeled[proj_labeled["Year"] == year_int]
               .groupby("Sex", as_index=False)["Child0_3_Buggy"].sum())
    fig_child_y = px.bar(
        child_y, x="Sex", y="Child0_3_Buggy", template="plotly_white",
        labels={"Child0_3_Buggy": "People (count)", "Sex": "Sex"},
        title=f"Children aged 0–3 (buggy) — {year_int}"
    )
    fig_child_y.update_yaxes(tickformat=",")
    st.plotly_chart(fig_child_y, use_container_width=True)

# -------------- Data table (filtered by year) ---------------
st.markdown("---")
st.subheader("Explore the data")

# Filter by selected year, sex, and band
if sel_year == "All Years":
    mask_year = proj_labeled["Year"].isin(years)
else:
    mask_year = proj_labeled["Year"] == int(sel_year)

mask = mask_year & proj_labeled["Sex"].isin(sel_sex) & proj_labeled["Band"].isin(sel_band)
show = proj_labeled[mask].copy().sort_values(["Year", "Sex", "Band"])

st.dataframe(
    show[["Year", "Sex", "Band", "ProjectedPopulation", "baseline_rate",
          "ProjectedStairDifficulty", "Child0_3_Buggy"]]
      .rename(columns={
          "ProjectedPopulation": "Projected population",
          "baseline_rate": "Baseline Q15(c) rate (decimal)",
          "ProjectedStairDifficulty": "People with difficulty (Q15c)",
          "Child0_3_Buggy": "Children aged 0–3 (buggy)"
      })
      .style.format({
          "Projected population": "{:,.0f}",
          "People with difficulty (Q15c)": "{:,.0f}",
          "Children aged 0–3 (buggy)": "{:,.0f}",
          "Baseline Q15(c) rate (decimal)": "{:.4f}",
      }),
    use_container_width=True
)