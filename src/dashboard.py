# ♿ Disability & Mobility Projections Dashboard
# Data: data/prep_PEC19.csv, data/baseline.csv, data/Households_size_2022.csv
# Note: We use baseline.csv (Band×Sex rates as decimals 0–1). raw_F4006.csv not required for runtime.

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------- Page config ----------------
st.set_page_config(page_title="Disability & Mobility Projections", page_icon="♿", layout="wide")

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
    hs = (hh["households_2022"] / hh["households_2022"].sum()).to_numpy()  # weights by size class
    s = hh["size_numeric"].to_numpy()                                       # sizes 1..7
    q_vals = q.to_numpy()[:, None]                                          # (n_years, 1)
    p_vals = (1.0 - (1.0 - q_vals) ** s) * hs                               # (n_years, n_sizes)
    return pd.Series(p_vals.sum(axis=1), index=q.index)

def to_title(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.title()

# -------------- Data load & prep ------------
@st.cache_data(show_spinner=True)
def load_inputs():
    # Population (PEC19)
    pop = pd.read_csv(DATA_DIR / "prep_PEC19.csv")
    pop.columns = [c.strip() for c in pop.columns]

    # Filter to Method - M2 if column exists
    if "Criteria for Projection" in pop.columns:
        m = pop["Criteria for Projection"].astype(str)
        pop = pop[m.str.contains("Method", na=False) & m.str.contains("M2", na=False)]

    # Standardize population column name to VALUE
    if "Population" in pop.columns and "VALUE" not in pop.columns:
        pop.rename(columns={"Population": "VALUE"}, inplace=True)
    if "VALUE" not in pop.columns:
        raise ValueError("prep_PEC19.csv must have 'VALUE' (or 'Population').")
    pop["VALUE"] = pd.to_numeric(pop["VALUE"], errors="coerce").fillna(0)

    # Create Band from AgeNum if needed
    if "Band" not in pop.columns:
        if "AgeNum" in pop.columns:
            pop["Band"] = band_from_age(pd.to_numeric(pop["AgeNum"], errors="coerce"))
        else:
            raise ValueError("prep_PEC19.csv needs 'Band' or 'AgeNum'.")

    # Keep only Male/Female
    pop["Sex"] = to_title(pop["Sex"])
    pop = pop[pop["Sex"].isin(["Male", "Female"])]

    # Focus years (if present)
    years_present = sorted(pop["Year"].unique().tolist())
    years = [y for y in YEARS_FOCUS if y in years_present] or years_present
    pop = pop[pop["Year"].isin(years)]

    # Aggregate to Year × Sex × Band
    pop_g = (pop.groupby(["Year", "Sex", "Band"], as_index=False)["VALUE"]
                .sum()
                .rename(columns={"VALUE": "ProjectedPopulation"}))

    # Baseline (Band×Sex rates, decimals)
    base = pd.read_csv(DATA_DIR / "baseline.csv")
    base.columns = [c.strip() for c in base.columns]
    if "baseline_rate" not in base.columns:
        if "Baseline %" in base.columns:
            base.rename(columns={"Baseline %": "baseline_rate"}, inplace=True)
        else:
            raise ValueError("baseline.csv must have 'baseline_rate' (decimal) or 'Baseline %'.")
    base["Sex"]  = to_title(base["Sex"])
    base["Band"] = base["Band"].astype(str).str.strip()
    base["baseline_rate"] = pd.to_numeric(base["baseline_rate"], errors="coerce")

    # Merge population with rates
    proj = pop_g.merge(base[["Band", "Sex", "baseline_rate"]],
                       on=["Band", "Sex"], how="left")
    if proj["baseline_rate"].isna().any():
        missing = proj[proj["baseline_rate"].isna()][["Band", "Sex"]].drop_duplicates()
        raise ValueError(f"Missing baseline_rate for Band×Sex combos:\n{missing}")

    # Expected persons with difficulty (incl. stairs)
    proj["ProjectedStairDifficulty"] = proj["ProjectedPopulation"] * proj["baseline_rate"]

    # IMPORTANT: ages 0–3 are NOT counted as 'stairs difficulty' persons (buggy proxy only)
    proj.loc[proj["Band"] == "0-3", "ProjectedStairDifficulty"] = 0.0

    # Child proxy for B (0–3)
    proj["ProjectedChildren0_3"] = np.where(
        proj["Band"] == "0-3", proj["ProjectedPopulation"], 0.0
    )

    # Household size mix (2022)
    hh = pd.read_csv(DATA_DIR / "Households_size_2022.csv")
    hh.columns = [c.strip() for c in hh.columns]
    needed = {"size_numeric", "households_2022"}
    if not needed.issubset(set(hh.columns)):
        raise ValueError("Households_size_2022.csv needs columns: size_numeric, households_2022")
    hh["households_2022"] = pd.to_numeric(hh["households_2022"], errors="coerce").fillna(0)
    hh_total = hh["households_2022"].sum()

    return proj, hh, hh_total, years

@st.cache_data(show_spinner=True)
def build_views(proj: pd.DataFrame, hh: pd.DataFrame):
    # Person totals & shares by year
    by_year = (proj.groupby("Year", as_index=False)
                  .agg(total_pop=("ProjectedPopulation", "sum"),
                       stairs=("ProjectedStairDifficulty", "sum"),
                       child03=("ProjectedChildren0_3", "sum")))
    by_year["q_stairs"]   = by_year["stairs"] / by_year["total_pop"]
    by_year["q_children"] = by_year["child03"] / by_year["total_pop"]

    # Household shares via size model
    by_year["pA_households"] = person_share_to_household_share(by_year["q_stairs"], hh)
    by_year["pB_households"] = person_share_to_household_share(by_year["q_children"], hh)
    by_year["p_union"]       = by_year["pA_households"] + by_year["pB_households"] - \
                               (by_year["pA_households"] * by_year["pB_households"])

    # Long labels for bands (for charts)
    band_long = {
        "0-3": "Ages 0–3 years",
        "4-64": "Ages 4–64 years",
        "65+": "Ages 65 years and over"
    }
    proj_lab = proj.copy()
    proj_lab["BandLabel"] = proj_lab["Band"].map(band_long)

    return by_year, proj_lab

# -------------- Load everything -------------
proj, hh, hh_total, years = load_inputs()
by_year, proj_labeled = build_views(proj, hh)

# ---------------- UI -----------------------
st.title("♿ Disability & Mobility Projections (Ireland)")
st.caption("Source: CSO PEC19 (Method M2), Census 2022, F4006 Q15(c).")

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

# -------------- Top KPIs -------------------
col1, col2, col3, col4 = st.columns(4)
yr2022 = 2022 if 2022 in years else years[0]
snap = proj_labeled[proj_labeled["Year"] == yr2022]
col1.metric(f"Total population ({yr2022})", f"{int(snap['ProjectedPopulation'].sum()):,}")
col2.metric("People with difficulty (incl. stairs)", f"{int(snap['ProjectedStairDifficulty'].sum()):,}")
col3.metric("Children aged 0–3 years", f"{int(snap['ProjectedChildren0_3'].sum()):,}")
h_row = by_year[by_year["Year"] == yr2022].iloc[0]
col4.metric("Households affected (either condition)", f"{100*h_row['p_union']:.2f}%")

st.markdown("---")

# -------------- All-years or single-year ---
if sel_year == "All Years":
    # Persons with difficulty — grouped+stacked by Year→Sex; stacks=age bands
    st.subheader("Projected individuals with difficulty (including stairs) — clustered by year and sex; stacks = age bands")
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

    # NEW: Children 0–3 (buggy proxy) — counts by year × sex
    st.subheader("Children aged 0–3 years (buggy proxy) — people (by year and sex)")
    child_viz = (proj_labeled
                 .groupby(["Year", "Sex"], as_index=False)["ProjectedChildren0_3"].sum())
    child_viz = child_viz[child_viz["Sex"].isin(sel_sex)].copy()
    child_viz["Year_str"] = child_viz["Year"].astype(str)
    fig_child = px.bar(
        child_viz,
        x=["Year_str", "Sex"], y="ProjectedChildren0_3",
        barmode="group", template="plotly_white",
        category_orders={"Year_str": [str(y) for y in years], "Sex": ["Male", "Female"]},
        labels={"Year_str": "Year", "Sex": "Sex", "ProjectedChildren0_3": "People (count)"},
        title="Year → Sex clusters (children aged 0–3)"
    )
    fig_child.update_yaxes(tickformat=",")
    st.plotly_chart(fig_child, use_container_width=True)

    # Household shares — lines (A, B, Union)
    st.subheader("Estimated share of households affected (percentages)")
    fig2 = px.line(
        by_year, x="Year", y=["pA_households", "pB_households", "p_union"],
        markers=True, template="plotly_white",
        labels={"value": "Share of households", "variable": "Measure"},
        title="Households with: (i) ≥1 member with difficulty (incl. stairs), (ii) ≥1 child aged 0–3, (iii) either condition"
    )
    for tr in fig2.data:  # convert to %
        tr.y = [100*v for v in tr.y]
    fig2.update_yaxes(title="Share of households (%)", rangemode="tozero")
    fig2.for_each_trace(lambda t: t.update(name={"pA_households": "Member with difficulty (incl. stairs)",
                                                 "pB_households": "Child aged 0–3",
                                                 "p_union": "Either condition (combined)"}[t.name]))
    st.plotly_chart(fig2, use_container_width=True)

    # Optional PNG export (requires kaleido)
    try:
        import kaleido  # noqa: F401
        fig1.write_image("outputs/persons_difficulty_grouped_stacked.png", scale=2, width=1200, height=600)
        fig_child.write_image("outputs/children03_grouped.png", scale=2, width=1200, height=500)
        fig2.write_image("outputs/households_shares_trend.png", scale=2, width=1200, height=500)
    except Exception:
        pass

else:
    year_int = int(sel_year)
    st.subheader(f"Detailed projections for {year_int}")

    df_y = proj_labeled[(proj_labeled["Year"] == year_int) &
                        (proj_labeled["Sex"].isin(sel_sex)) &
                        (proj_labeled["Band"].isin(sel_band))].copy()

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("People with difficulty (incl. stairs)", f"{int(df_y['ProjectedStairDifficulty'].sum()):,}")
    c2.metric("Children aged 0–3 years (buggy proxy)", f"{int(df_y['ProjectedChildren0_3'].sum()):,}")
    h_row = by_year[by_year["Year"] == year_int].iloc[0]
    c3.metric("Households affected — either condition", f"{100*h_row['p_union']:.2f}%")

    # Persons with difficulty — grouped by age band × sex
    fig_bar = px.bar(
        df_y, x="Band", y="ProjectedStairDifficulty", color="Sex",
        barmode="group", template="plotly_white",
        category_orders={"Band": ["0-3", "4-64", "65+"], "Sex": ["Male", "Female"]},
        labels={"Band": "Age band", "ProjectedStairDifficulty": "People (count)"},
        title=f"Individuals with difficulty (incl. stairs), by age band and sex — {year_int}"
    )
    fig_bar.update_yaxes(tickformat=",")
    st.plotly_chart(fig_bar, use_container_width=True)

    # NEW: Children 0–3 by sex for the selected year
    child_y = (proj_labeled[(proj_labeled["Year"] == year_int)]
               .groupby("Sex", as_index=False)["ProjectedChildren0_3"].sum())
    fig_child_y = px.bar(
        child_y, x="Sex", y="ProjectedChildren0_3", template="plotly_white",
        labels={"ProjectedChildren0_3": "People (count)", "Sex": "Sex"},
        title=f"Children aged 0–3 years (buggy proxy) — {year_int}"
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
          "ProjectedStairDifficulty", "ProjectedChildren0_3"]]
      .rename(columns={
          "ProjectedPopulation": "Projected population",
          "ProjectedStairDifficulty": "People with difficulty (incl. stairs)",
          "ProjectedChildren0_3": "Children aged 0–3",
          "baseline_rate": "Baseline rate (decimal)"
      })
      .style.format({
          "Projected population": "{:,.0f}",
          "People with difficulty (incl. stairs)": "{:,.0f}",
          "Children aged 0–3": "{:,.0f}",
          "Baseline rate (decimal)": "{:.4f}",
      }),
    use_container_width=True
)