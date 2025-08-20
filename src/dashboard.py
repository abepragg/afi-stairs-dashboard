# Save this code in: afi-stairs-dashboard/src/dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Disability & Mobility Projections",
    page_icon="♿",
    layout="wide"
)

# --- Data Loading and Caching ---
@st.cache_data
def load_and_prepare_data():
    """
    Loads, preprocesses, and merges all necessary data from CSV files.
    This version dynamically calculates baseline disability rates from 'raw_F4006.csv'.
    """
    try:
        path_to_data = 'data/'
        population_projections = pd.read_csv(os.path.join(path_to_data, 'prep_PEC19.csv'))
        raw_disability_data = pd.read_csv(os.path.join(path_to_data, 'raw_F4006.csv'))
        households_df = pd.read_csv(os.path.join(path_to_data, 'Households_size_2022.csv'))

        # --- 1. Calculate 2022 Baseline Rates from Raw Data ---
        pop_2022_df = population_projections[population_projections['Year'] == 2022].copy()
        total_pop_2022_by_group = pop_2022_df.groupby(['Band', 'Sex'])['VALUE'].sum().reset_index()
        total_pop_2022_by_group.rename(columns={'VALUE': 'TotalPopulation2022'}, inplace=True)

        any_disability_2022_df = raw_disability_data[raw_disability_data['STATISTIC'] == 'F4006C01']
        total_any_disability_by_group = any_disability_2022_df.groupby(['Band', 'Sex'])['VALUE'].sum().reset_index()
        total_any_disability_by_group.rename(columns={'VALUE': 'AnyDisabilityCount2022'}, inplace=True)
        
        stair_difficulty_codes = ['F4006C12', 'F4006C13']
        stair_difficulty_2022_df = raw_disability_data[raw_disability_data['STATISTIC'].isin(stair_difficulty_codes)]
        total_stair_difficulty_by_group = stair_difficulty_2022_df.groupby(['Band', 'Sex'])['VALUE'].sum().reset_index()
        total_stair_difficulty_by_group.rename(columns={'VALUE': 'StairDifficultyCount2022'}, inplace=True)

        baseline_rates = pd.merge(total_pop_2022_by_group, total_any_disability_by_group, on=['Band', 'Sex'])
        baseline_rates = pd.merge(baseline_rates, total_stair_difficulty_by_group, on=['Band', 'Sex'])
        
        baseline_rates['AnyDisabilityRate'] = baseline_rates['AnyDisabilityCount2022'] / baseline_rates['TotalPopulation2022']
        baseline_rates['StairDifficultyRate'] = baseline_rates['StairDifficultyCount2022'] / baseline_rates['TotalPopulation2022']

        # --- 2. Prepare Projections for all years (including 2022) ---
        projections_df = population_projections[population_projections['Year'].isin([2022, 2030, 2040, 2050])].copy()
        projections_df_agg = projections_df.groupby(['Year', 'Band', 'Sex'])['VALUE'].sum().reset_index()
        projections_df_agg.rename(columns={'VALUE': 'ProjectedPopulation'}, inplace=True)

        merged_df = pd.merge(
            projections_df_agg,
            baseline_rates[['Band', 'Sex', 'AnyDisabilityRate', 'StairDifficultyRate']],
            on=['Band', 'Sex'],
            how='left'
        )

        merged_df['ProjectedAnyDisability'] = merged_df['ProjectedPopulation'] * merged_df['AnyDisabilityRate']
        merged_df['ProjectedStairDifficulty'] = merged_df['ProjectedPopulation'] * merged_df['StairDifficultyRate']
        
        is_0_3_band = (merged_df['Band'] == '0-3')
        merged_df.loc[is_0_3_band, 'ProjectedStairDifficulty'] = merged_df.loc[is_0_3_band, 'ProjectedPopulation']

        total_households_2022 = households_df['households_2022'].sum()

        return merged_df, total_households_2022

    except Exception as e:
        st.error(f"An error occurred during data loading. Details: {e}")
        st.stop()


# --- Main Application ---
df, total_households = load_and_prepare_data()

st.title("♿ Disability & Mobility Projections for Ireland")

# --- STATIC 2022 BASELINE SNAPSHOT ---
st.header("Baseline: Census 2022 Snapshot")
data_2022 = df[df['Year'] == 2022]
total_pop_2022 = data_2022['ProjectedPopulation'].sum()
total_any_disability_2022 = data_2022['ProjectedAnyDisability'].sum()
total_stair_difficulty_2022 = data_2022['ProjectedStairDifficulty'].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total Population (2022)", f"{int(total_pop_2022):,}")
col2.metric("Total with ANY Disability", f"{int(total_any_disability_2022):,}")
col3.metric("Total with Stair Difficulty", f"{int(total_stair_difficulty_2022):,}")
st.markdown("---")

# --- FILTERS AND DYNAMIC CONTENT ---
st.sidebar.header("Dashboard Filters")
year_options = ["All Years"] + sorted(df['Year'].unique())
selected_year = st.sidebar.selectbox('Select Year', options=year_options)
selected_sex = st.sidebar.multiselect('Select Gender', options=df['Sex'].unique(), default=df['Sex'].unique())
selected_band = st.sidebar.multiselect('Select Age Band', options=df['Band'].unique(), default=df['Band'].unique())

if not selected_sex or not selected_band:
    st.warning("Please select at least one gender and one age band.")
    st.stop()

# --- DYNAMIC DISPLAY AREA ---
if selected_year == "All Years":
    st.header("Trend Projections: 2022-2050")
    filtered_df = df[(df['Sex'].isin(selected_sex)) & (df['Band'].isin(selected_band))]
    
    # Chart 1: Percentage of Population with Stair Difficulty Over Time
    trend_rate_df = filtered_df.groupby('Year').agg(
        TotalPop=('ProjectedPopulation', 'sum'),
        StairDiff=('ProjectedStairDifficulty', 'sum')
    ).reset_index()
    trend_rate_df['Rate'] = (trend_rate_df['StairDiff'] / trend_rate_df['TotalPop']) * 100
    fig_rate_trend = px.line(
        trend_rate_df, x='Year', y='Rate',
        title='Percentage of Total Population with Stair Difficulty',
        labels={'Rate': 'Percentage (%)'}, markers=True
    )
    # This line saves the chart as a high-quality PNG image file
    fig_rate_trend.write_image("trend_percent_of_population.png", width=1000, height=500, scale=2)
    st.plotly_chart(fig_rate_trend, use_container_width=True)
    
    # Chart 2: Composition of the Stair Difficulty Group Over Time
    composition_df = filtered_df.groupby(['Year', 'Band'])['ProjectedStairDifficulty'].sum().reset_index()
    fig_composition = px.area(
        composition_df, x='Year', y='ProjectedStairDifficulty', color='Band',
        title='Changing Composition of the Stair Difficulty Population',
        labels={'ProjectedStairDifficulty': 'Number of Individuals'},
        category_orders={"Band": ["0-3", "4-64", "65+"]}
    )
    # This line saves the chart as a high-quality PNG image file
    fig_composition.write_image("trend_composition_by_age.png", width=1000, height=500, scale=2)
    st.plotly_chart(fig_composition, use_container_width=True)
    
else: # A single year is selected
    st.header(f"Detailed Projections for {selected_year}")
    filtered_df = df[
        (df['Year'] == selected_year) & (df['Sex'].isin(selected_sex)) & (df['Band'].isin(selected_band))
    ]
    
    pop_year = filtered_df['ProjectedPopulation'].sum()
    any_disability_year = filtered_df['ProjectedAnyDisability'].sum()
    stair_difficulty_year = filtered_df['ProjectedStairDifficulty'].sum()
    
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Total Projected Population ({selected_year})", f"{int(pop_year):,}")
    c2.metric("Projected with ANY Disability", f"{int(any_disability_year):,}")
    c3.metric("Projected with Stair Difficulty", f"{int(stair_difficulty_year):,}")

    st.write(f"""
    In **{selected_year}**, an estimated **{stair_difficulty_year / pop_year:.2%}** of the total projected population will have a stair-climbing difficulty. 
    This group is projected to represent **{stair_difficulty_year / any_disability_year:.2%}** of all individuals with any form of disability.
    """)
    
    children_3_and_under = int(filtered_df[filtered_df['Band'] == '0-3']['ProjectedStairDifficulty'].sum())
    people_with_stair_issues = int(filtered_df[filtered_df['Band'] != '0-3']['ProjectedStairDifficulty'].sum())
    household_proxy = ((children_3_and_under + people_with_stair_issues) / total_households) * 100
    st.metric("Percentage of 2022 Households with an Affected Member (Proxy)", f"{household_proxy:.2f}%")
    
    st.subheader("Stair Difficulty as a Percentage of Total Disability (by Age Band)")
    comparison_df = filtered_df[filtered_df['Band'] != '0-3'].groupby('Band').agg({
        'ProjectedAnyDisability': 'sum', 'ProjectedStairDifficulty': 'sum'
    }).reset_index()
    comparison_df['Stair_as_pct_of_Any'] = (comparison_df['ProjectedStairDifficulty'] / comparison_df['ProjectedAnyDisability']) * 100
    
    col1, col2 = st.columns(2)
    for band, col in zip(["4-64", "65+"], [col1, col2]):
        with col:
            band_data = comparison_df[comparison_df['Band'] == band]
            if not band_data.empty:
                val = band_data['Stair_as_pct_of_Any'].iloc[0]
                st.metric(f"{band} Age Band", f"{val:.1f}%")
    st.caption("Note: The 0-3 age band is excluded from this comparison due to the 'buggy rule' assumption.")
    
    # Chart 3: Bar Chart for single-year view
    fig_bar = px.bar(
        filtered_df, x='Band', y='ProjectedStairDifficulty', color='Sex', barmode='group',
        title=f'Projected Individuals with Stair Difficulty by Age Band & Gender ({selected_year})',
        labels={'ProjectedStairDifficulty': 'Number of Individuals', 'Band': 'Age Band'},
        color_discrete_map={'Male': '#1f77b4', 'Female': '#ff7f0e'},
        category_orders={"Band": ["0-3", "4-64", "65+"]}
    )
    # This line saves the chart as a high-quality PNG image file
    fig_bar.write_image("single_year_bar_chart.png", width=1000, height=500, scale=2)
    st.plotly_chart(fig_bar, use_container_width=True)

# --- Explore Data Table ---
st.header("Explore the Data")
display_df = filtered_df.groupby(['Year', 'Sex', 'Band']).agg(
    ProjectedPopulation=('ProjectedPopulation', 'sum'),
    ProjectedStairDifficulty=('ProjectedStairDifficulty', 'sum'),
    StairDifficultyRate=('StairDifficultyRate', 'first') 
).reset_index()
display_df.loc[display_df['Band'] == '0-3', 'StairDifficultyRate'] = 1.0

st.dataframe(
    display_df[['Year', 'Sex', 'Band', 'ProjectedPopulation', 'StairDifficultyRate', 'ProjectedStairDifficulty']]
    .style.format({
        'ProjectedPopulation': '{:,.0f}', 'ProjectedStairDifficulty': '{:,.0f}', 'StairDifficultyRate': '{:.2%}'
    })
)