import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime

# Set up page configurations
st.set_page_config(
    page_title="COVID-19 Global Analytics Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (custom cards, gradients, borders)
st.markdown("""
<style>
    /* Main body background & custom fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Title text styling */
    .title-gradient {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
        margin-top: -1rem;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Premium Metric Card design */
    .metric-container {
        display: flex;
        gap: 15px;
        flex-wrap: wrap;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.15);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        flex: 1;
        min-width: 220px;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.1);
        background: rgba(30, 41, 59, 0.6);
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    
    .metric-change {
        font-size: 0.82rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    
    .change-good {
        color: #10b981; /* green */
    }
    
    .change-bad {
        color: #f43f5e; /* rose/red */
    }
    
    .change-neutral {
        color: #94a3b8; /* gray */
    }
    
    /* Country Profile Info Card */
    .profile-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 14px;
        padding: 24px;
        color: #e2e8f0;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }
    
    .profile-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 10px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .profile-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 20px;
    }
    
    .profile-item {
        border-left: 3px solid #818cf8;
        padding-left: 12px;
    }
    
    .profile-label {
        font-size: 0.72rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 3px;
        font-weight: 500;
    }
    
    .profile-val {
        font-size: 1.15rem;
        font-weight: 600;
        color: #ffffff;
    }
    
    /* Alert details */
    .info-alert {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 8px;
        padding: 12px 16px;
        color: #c7d2fe;
        font-size: 0.9rem;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to format big numbers cleanly
def format_num(val):
    if val is None:
        return "N/A"
    try:
        val_f = float(val)
        if val_f >= 1_000_000_000:
            return f"{val_f / 1_000_000_000:.2f}B"
        elif val_f >= 1_000_000:
            return f"{val_f / 1_000_000:.2f}M"
        elif val_f >= 1_000:
            return f"{val_f:,.0f}"
        elif val_f == 0:
            return "0"
        elif abs(val_f) < 0.01:
            return f"{val_f:.6f}"
        elif abs(val_f) < 1.0:
            return f"{val_f:.4f}"
        return f"{val_f:.2f}"
    except (ValueError, TypeError):
        return str(val)

# Dynamic cached loading of compact.csv via Polars (very fast)
@st.cache_data
def load_data():
    df = pl.read_csv("compact.csv", try_parse_dates=True)
    # Filter rows with null dates and sort ascending
    df = df.filter(pl.col("date").is_not_null()).sort("date")
    return df

# Main data loading wrapper
try:
    with st.spinner("⚡ Loading and indexing COVID-19 dataset with Polars..."):
        df = load_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.info("Please ensure compact.csv is placed in the workspace directory.")
    st.stop()

# Basic Date boundaries from Polars DataFrame
min_date = df["date"].min()
max_date = df["date"].max()

# Meta Dictionary organizing metrics into visual categories
METRICS_MAP = {
    "📊 Cases": {
        "New Cases": "new_cases",
        "New Cases (Smoothed)": "new_cases_smoothed",
        "Total Cases": "total_cases",
        "New Cases per Million": "new_cases_per_million",
        "New Cases (Smoothed) per Million": "new_cases_smoothed_per_million",
        "Total Cases per Million": "total_cases_per_million",
    },
    "☠️ Deaths": {
        "New Deaths": "new_deaths",
        "New Deaths (Smoothed)": "new_deaths_smoothed",
        "Total Deaths": "total_deaths",
        "New Deaths per Million": "new_deaths_per_million",
        "New Deaths (Smoothed) per Million": "new_deaths_smoothed_per_million",
        "Total Deaths per Million": "total_deaths_per_million",
    },
    "💉 Vaccinations": {
        "New Vaccinations": "new_vaccinations",
        "New Vaccinations (Smoothed)": "new_vaccinations_smoothed",
        "Total Vaccinations": "total_vaccinations",
        "People Vaccinated": "people_vaccinated",
        "People Fully Vaccinated": "people_fully_vaccinated",
        "Total Boosters": "total_boosters",
        "Total Vaccinations per Hundred": "total_vaccinations_per_hundred",
        "People Vaccinated per Hundred": "people_vaccinated_per_hundred",
        "People Fully Vaccinated per Hundred": "people_fully_vaccinated_per_hundred",
    },
    "🏥 Hospitalizations & ICU": {
        "Hosp Patients": "hosp_patients",
        "Hosp Patients per Million": "hosp_patients_per_million",
        "ICU Patients": "icu_patients",
        "ICU Patients per Million": "icu_patients_per_million",
        "Weekly Hosp Admissions": "weekly_hosp_admissions",
        "Weekly ICU Admissions": "weekly_icu_admissions",
    },
    "🧪 Testing": {
        "New Tests": "new_tests",
        "Total Tests": "total_tests",
        "Positive Rate": "positive_rate",
        "Tests per Case": "tests_per_case",
    },
    "🛡️ Policies & Other": {
        "Stringency Index": "stringency_index",
        "Reproduction Rate": "reproduction_rate",
        "Excess Mortality": "excess_mortality",
        "Excess Mortality Cumulative": "excess_mortality_cumulative",
    }
}

# Sidebar configuration
st.sidebar.markdown("### 🛠️ Configuration")

# Dynamic metric selector
metric_category = st.sidebar.selectbox("Metric Category", list(METRICS_MAP.keys()))
metric_label = st.sidebar.selectbox("Change Metric To View", list(METRICS_MAP[metric_category].keys()))
metric_col = METRICS_MAP[metric_category][metric_label]

# Dynamic date range selector
st.sidebar.markdown("### 📅 Date Filter")
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Parse date range input safely (in case user is click-dragging)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# Sidebar stats summary
total_rows = len(df)
st.sidebar.markdown(f"""
---
**Dataset Information**
* Total Records: `{total_rows:,}`
* Start Date: `{min_date}`
* End Date: `{max_date}`
* Powered by **Polars** & **Streamlit**
""")

# Main Header Area
st.markdown('<div class="title-gradient">COVID-19 Global Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An interactive, high-performance visualization dashboard powered by Polars dataframes.</div>', unsafe_allow_html=True)

# Split database into Aggregates and Actual Countries
# Actual countries are rows where `continent` is not null
countries_df = df.filter(pl.col("continent").is_not_null())
country_list = sorted(countries_df["country"].unique().to_list())

# Aggregate global data row filter
global_df = df.filter(pl.col("country") == "World")

# Helper function to generate premium metric card HTML
def make_metric_card(label, latest_val, prev_val, is_good_increase=False):
    val_str = format_num(latest_val)
    
    if latest_val is not None and prev_val is not None:
        try:
            diff = float(latest_val) - float(prev_val)
            pct = (diff / float(prev_val) * 100) if float(prev_val) != 0 else 0
            
            is_pos = diff > 0
            if diff == 0:
                change_class = "change-neutral"
                change_str = "No Change"
            else:
                # Decide if change is good or bad
                if is_pos:
                    change_class = "change-bad" if not is_good_increase else "change-good"
                    change_str = f"▲ {format_num(diff)} ({pct:+.1f}%)"
                else:
                    change_class = "change-good" if not is_good_increase else "change-bad"
                    change_str = f"▼ {format_num(abs(diff))} ({pct:+.1f}%)"
        except (ValueError, TypeError):
            change_class = "change-neutral"
            change_str = "WoW N/A"
    else:
        change_class = "change-neutral"
        change_str = "WoW N/A"
        
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{val_str}</div>
        <div class="metric-change {change_class}">{change_str} <span style="font-size: 0.72rem; color: #94a3b8; font-weight: normal; margin-left: 2px;">WoW</span></div>
    </div>
    """

# Create Tabs
tab_global, tab_country, tab_compare, tab_explorer = st.tabs([
    "🌎 Global Analytics",
    "📍 Country Profiles",
    "📊 Multi-Country Comparison",
    "🔍 Data Search & Explorer"
])

# ----------------- TAB 1: GLOBAL ANALYTICS -----------------
with tab_global:
    st.markdown("### 🌎 Worldwide Covid-19 Status")
    
    # Extract latest global stats
    latest_global_df = global_df.filter(pl.col(metric_col).is_not_null()).sort("date", descending=True)
    
    if len(latest_global_df) > 0:
        latest_row = latest_global_df.head(1).to_dicts()[0]
        latest_date_available = latest_row["date"]
        
        # Fetch WoW (7 days ago) row
        wow_global_df = latest_global_df.filter(pl.col("date") <= latest_date_available - pl.duration(days=7)).sort("date", descending=True)
        wow_row = wow_global_df.head(1).to_dicts()[0] if len(wow_global_df) > 0 else {}
        
        # Fetch core global values for card grid
        cases_now = latest_row.get("total_cases")
        cases_prev = wow_row.get("total_cases")
        
        deaths_now = latest_row.get("total_deaths")
        deaths_prev = wow_row.get("total_deaths")
        
        vax_now = latest_row.get("total_vaccinations")
        vax_prev = wow_row.get("total_vaccinations")
        
        selected_now = latest_row.get(metric_col)
        selected_prev = wow_row.get(metric_col)
        
        # Decide color coding for chosen metric
        is_metric_good_increase = any(x in metric_col for x in ["vaccin", "test", "beds", "expectancy", "hdi"])
        
        # Render cards side by side
        st.markdown(f"""
        <div class="metric-container">
            {make_metric_card("Total Cases (Global)", cases_now, cases_prev, False)}
            {make_metric_card("Total Deaths (Global)", deaths_now, deaths_prev, False)}
            {make_metric_card("Total Vaccinations (Global)", vax_now, vax_prev, True)}
            {make_metric_card(f"{metric_label} (Global)", selected_now, selected_prev, is_metric_good_increase)}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No global data records found for the selected metric.")

    # Global Trend over Selected Date Range
    st.markdown("#### Global Trend Line")
    filtered_global = global_df.filter(
        (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
    ).filter(pl.col(metric_col).is_not_null())
    
    if len(filtered_global) > 0:
        fig_global = px.line(
            filtered_global.to_pandas(),
            x="date",
            y=metric_col,
            labels={"date": "Date", metric_col: metric_label}
        )
        fig_global.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=10, b=20),
            hovermode="x unified",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            height=380
        )
        fig_global.update_traces(line_color="#6366f1", line_width=3)
        st.plotly_chart(fig_global, use_container_width=True)
    else:
        st.info("No global data points match the chosen date range.")
        
    st.markdown("---")
    
    # Map & Continent Breakdown columns
    col_map, col_continent = st.columns([3, 2])
    
    # 1. Fetch latest record for each actual country
    # We sort by date ascending so that .group_by("country").last() pulls the latest date row for each country
    latest_countries = (
        countries_df.filter(pl.col(metric_col).is_not_null())
        .sort("date")
        .group_by("country")
        .last()
    )
    
    with col_map:
        st.markdown("#### 🗺️ Global Distribution Map")
        if len(latest_countries) > 0:
            fig_map = px.choropleth(
                latest_countries.to_pandas(),
                locations="code",
                color=metric_col,
                hover_name="country",
                hover_data={"code": False, "continent": True, metric_col: ":,.2f"},
                color_continuous_scale=px.colors.sequential.Plasma,
                labels={metric_col: metric_label}
            )
            fig_map.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=20, b=0),
                height=450,
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    coastlinecolor="rgba(255,255,255,0.15)",
                    projection_type="natural earth",
                    bgcolor="rgba(0,0,0,0)",
                    landcolor="#1e293b",
                    lakecolor="#0f172a"
                ),
                coloraxis_colorbar=dict(
                    title="",
                    thicknessmode="pixels", thickness=15,
                    lenmode="pixels", len=200,
                    yanchor="bottom", y=0.1
                )
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No country distribution data available for the map.")
            
    with col_continent:
        st.markdown("#### 📊 Regional Comparison")
        if len(latest_countries) > 0:
            # Determine appropriate aggregation type (sum for counts, mean for rates)
            is_rate = any(x in metric_col for x in ["per_million", "per_thousand", "per_hundred", "rate", "index", "prevalence", "expectancy", "age"])
            agg_expr = pl.col(metric_col).mean().alias("val") if is_rate else pl.col(metric_col).sum().alias("val")
            
            continent_summary = (
                latest_countries.group_by("continent")
                .agg(agg_expr)
                .filter(pl.col("continent").is_not_null())
                .sort("val", descending=True)
            )
            
            fig_continent = px.bar(
                continent_summary.to_pandas(),
                x="continent",
                y="val",
                color="val",
                color_continuous_scale=px.colors.sequential.Viridis,
                labels={"continent": "Continent", "val": metric_label}
            )
            fig_continent.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=20, b=20),
                height=450,
                xaxis=dict(title=""),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_continent, use_container_width=True)
        else:
            st.info("No data available to create continent comparison.")

# ----------------- TAB 2: COUNTRY PROFILES -----------------
with tab_country:
    st.markdown("### 📍 Country Profile & Trend Analytics")
    
    # Country drop-down filter
    selected_country = st.selectbox(
        "Choose Country to Profile",
        options=country_list,
        index=country_list.index("United States") if "United States" in country_list else 0
    )
    
    # Filter country-specific Polars DataFrame
    country_df = df.filter(pl.col("country") == selected_country)
    
    # Grab static demographic data from the first row of country dataframe
    first_country_row = country_df.head(1).to_dicts()
    if first_country_row:
        prof = first_country_row[0]
        
        # Display custom styled demographic grid
        st.markdown(f"""
        <div class="profile-card">
            <div class="profile-header">
                📍 Demographics & Country Facts: <b>{selected_country}</b>
            </div>
            <div class="profile-grid">
                <div class="profile-item">
                    <div class="profile-label">Population</div>
                    <div class="profile-val">{format_num(prof.get("population"))}</div>
                </div>
                <div class="profile-item">
                    <div class="profile-label">GDP Per Capita</div>
                    <div class="profile-val">{'$' + format_num(prof.get("gdp_per_capita")) if prof.get("gdp_per_capita") is not None else 'N/A'}</div>
                </div>
                <div class="profile-item">
                    <div class="profile-label">Median Age</div>
                    <div class="profile-val">{f"{prof.get('median_age'):.1f} yrs" if prof.get("median_age") is not None else 'N/A'}</div>
                </div>
                <div class="profile-item">
                    <div class="profile-label">Life Expectancy</div>
                    <div class="profile-val">{f"{prof.get('life_expectancy'):.1f} yrs" if prof.get("life_expectancy") is not None else 'N/A'}</div>
                </div>
                <div class="profile-item">
                    <div class="profile-label">Hospital Beds</div>
                    <div class="profile-val">{f"{prof.get('hospital_beds_per_thousand'):.2f} / 1k" if prof.get("hospital_beds_per_thousand") is not None else 'N/A'}</div>
                </div>
                <div class="profile-item">
                    <div class="profile-label">HDI</div>
                    <div class="profile-val">{f"{prof.get('human_development_index'):.3f}" if prof.get("human_development_index") is not None else 'N/A'}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Create side-by-side charts: Specific Country trend vs Global Average comparison
    col_ctrend, col_ccomp = st.columns(2)
    
    # Filter country data by date range
    filtered_country = country_df.filter(
        (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
    ).filter(pl.col(metric_col).is_not_null())
    
    with col_ctrend:
        st.markdown(f"#### {selected_country} - {metric_label} Trend")
        if len(filtered_country) > 0:
            fig_ctrend = px.line(
                filtered_country.to_pandas(),
                x="date",
                y=metric_col,
                labels={"date": "Date", metric_col: metric_label}
            )
            fig_ctrend.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=10, b=20),
                hovermode="x unified",
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                height=350
            )
            fig_ctrend.update_traces(line_color="#a855f7", line_width=2.5)
            st.plotly_chart(fig_ctrend, use_container_width=True)
        else:
            st.info(f"No {metric_label} data points matching active date range for {selected_country}.")
            
    with col_ccomp:
        st.markdown(f"#### Comparison: {selected_country} vs. Global Average")
        
        # Prepare country and world series
        country_series = country_df.select(["date", metric_col]).rename({metric_col: selected_country})
        world_series = global_df.select(["date", metric_col]).rename({metric_col: "Global Average"})
        
        # Join on date using Polars (very fast)
        comparison_series = country_series.join(world_series, on="date", how="inner").filter(
            (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
        )
        
        if len(comparison_series) > 0:
            fig_comp = px.line(
                comparison_series.to_pandas(),
                x="date",
                y=[selected_country, "Global Average"],
                labels={"date": "Date", "value": metric_label, "variable": "Region"}
            )
            fig_comp.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=10, b=20),
                hovermode="x unified",
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            # Custom line colors
            color_map = {selected_country: "#a855f7", "Global Average": "#94a3b8"}
            for idx, trace in enumerate(fig_comp.data):
                trace.line.color = color_map.get(trace.name, "#6366f1")
                trace.line.width = 2.5
                
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("No comparative trend data available for this range.")

# ----------------- TAB 3: MULTI-COUNTRY COMPARISON -----------------
with tab_compare:
    st.markdown("### 📊 Compare Multiple Countries")
    st.markdown("Select multiple countries to overlay their metrics and compare the outbreak size, vaccination rollouts, or hospitalizations directly.")
    
    # Multiselect list of countries
    default_countries = ["United States", "United Kingdom", "Canada", "Germany", "France"]
    valid_defaults = [c for c in default_countries if c in country_list]
    
    selected_compare = st.multiselect(
        "Choose Countries to Overlay",
        options=country_list,
        default=valid_defaults[:3] if valid_defaults else country_list[:2]
    )
    
    if selected_compare:
        # Filter matching rows via Polars
        compare_df = df.filter(
            (pl.col("country").is_in(selected_compare)) &
            (pl.col("date") >= start_date) &
            (pl.col("date") <= end_date)
        ).filter(pl.col(metric_col).is_not_null())
        
        if len(compare_df) > 0:
            fig_compare = px.line(
                compare_df.to_pandas(),
                x="date",
                y=metric_col,
                color="country",
                labels={"date": "Date", metric_col: metric_label, "country": "Country"}
            )
            fig_compare.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=10, b=20),
                hovermode="x unified",
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                height=500
            )
            st.plotly_chart(fig_compare, use_container_width=True)
        else:
            st.warning("No valid data available for selected countries within this date range.")
    else:
        st.info("Please select one or more countries in the dropdown above to display the comparison chart.")

# ----------------- TAB 4: DATA EXPLORER & SEARCH -----------------
with tab_explorer:
    st.markdown("### 🔍 Search & Export Datasets")
    st.markdown("Search records using fast Polars processing and export custom chunks as CSV.")
    
    # Layout filter boxes
    col_exp_c, col_exp_d = st.columns(2)
    
    with col_exp_c:
        explore_countries = st.multiselect(
            "Filter Explorer by Country/Region",
            options=sorted(df["country"].unique().to_list()),
            key="exp_countries"
        )
        
    with col_exp_d:
        explore_dates = st.date_input(
            "Filter Explorer by Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="exp_dates"
        )
        
    # Process explorer dates range
    if isinstance(explore_dates, tuple) and len(explore_dates) == 2:
        exp_start, exp_end = explore_dates
    else:
        exp_start, exp_end = min_date, max_date
        
    # Apply filtering sequence with Polars
    filtered_exp_df = df
    if explore_countries:
        filtered_exp_df = filtered_exp_df.filter(pl.col("country").is_in(explore_countries))
    
    filtered_exp_df = filtered_exp_df.filter(
        (pl.col("date") >= exp_start) & (pl.col("date") <= exp_end)
    )
    
    # Allow column selection
    st.markdown("#### Select Columns to Display")
    display_columns = st.multiselect(
        "Columns",
        options=df.columns,
        default=["country", "date", "new_cases", "total_cases", "new_deaths", "total_deaths", "people_vaccinated"]
    )
    
    if not display_columns:
        display_columns = ["country", "date"]
        
    # Check matching count
    row_count = len(filtered_exp_df)
    st.markdown(f"Found **{row_count:,}** matching records. Displaying top 1,000 below.")
    
    if row_count > 0:
        # Show top 1000 records in table
        st.dataframe(
            filtered_exp_df.select(display_columns).head(1000).to_pandas(),
            use_container_width=True
        )
        
        # Download action (converts selected data to CSV)
        csv_payload = filtered_exp_df.select(display_columns).to_pandas().to_csv(index=False)
        st.download_button(
            label="💾 Download Filtered Dataset as CSV",
            data=csv_payload,
            file_name="covid_dashboard_export.csv",
            mime="text/csv",
            help="Downloads the entire filtered set (not just the top 1,000 lines)."
        )
    else:
        st.info("No matching records found for active filters.")
