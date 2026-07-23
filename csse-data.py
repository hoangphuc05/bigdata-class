import streamlit as st
import duckdb
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date

# 1. Page Configuration & Custom Aesthetics
st.set_page_config(
    page_title="CSSE COVID-19 Global & US Intelligence Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS styling (glassmorphic cards, custom gradients, dark mode polish)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .title-gradient {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        margin-top: -1rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Metric Cards Styling */
    .metric-container {
        display: flex;
        gap: 15px;
        flex-wrap: wrap;
        margin-bottom: 25px;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        flex: 1;
        min-width: 200px;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
        background: rgba(30, 41, 59, 0.75);
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
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    
    .metric-sub {
        font-size: 0.8rem;
        color: #cbd5e1;
    }

    /* Info Badge Box */
    .info-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 20px;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to format numbers cleanly
def format_num(val):
    if val is None or pd.isna(val):
        return "N/A"
    try:
        val_f = float(val)
        if abs(val_f) >= 1_000_000_000:
            return f"{val_f / 1_000_000_000:.2f}B"
        elif abs(val_f) >= 1_000_000:
            return f"{val_f / 1_000_000:.2f}M"
        elif abs(val_f) >= 1_000:
            return f"{val_f:,.0f}"
        elif val_f == 0:
            return "0"
        elif abs(val_f) < 0.01:
            return f"{val_f:.4f}"
        else:
            return f"{val_f:.2f}"
    except (ValueError, TypeError):
        return str(val)

# 2. Database Connection & Schema Initialization
@st.cache_resource
def get_db_connection():
    csv_path_us = "./CSSE-data/csse_covid_19_data/csse_covid_19_daily_reports_us/*.csv"
    csv_path_global = "./CSSE-data/csse_covid_19_data/csse_covid_19_daily_reports/*.csv"
    
    conn = duckdb.connect(database='test.duckdb', read_only=False)
    
    # Create clean views for Global & US CSSE daily reports
    conn.execute(f"""
    CREATE OR REPLACE VIEW clean_world_reports AS
    SELECT 
        strptime(regexp_extract(filename, '(\\d{{2}}-\\d{{2}}-\\d{{4}})', 1), '%m-%d-%Y')::DATE AS report_date,
        COALESCE(Country_Region, "Country/Region") AS country,
        COALESCE(Province_State, "Province/State") AS province_state,
        FIPS,
        Admin2,
        COALESCE(Lat, Latitude) AS lat,
        COALESCE(Long_, Longitude) AS lon,
        COALESCE(Confirmed, 0) AS confirmed,
        COALESCE(Deaths, 0) AS deaths,
        COALESCE(Recovered, 0) AS recovered,
        COALESCE(Active, 0) AS active,
        Combined_Key,
        COALESCE(Incident_Rate, Incidence_Rate) AS incident_rate,
        COALESCE(Case_Fatality_Ratio, "Case-Fatality_Ratio") AS case_fatality_ratio
    FROM read_csv_auto('{csv_path_global}', union_by_name=true, filename=true);
    """)

    conn.execute(f"""
    CREATE OR REPLACE VIEW clean_us_reports AS
    SELECT 
        COALESCE(Date, strptime(regexp_extract(filename, '(\\d{{2}}-\\d{{2}}-\\d{{4}})', 1), '%m-%d-%Y')::DATE) AS report_date,
        Province_State AS province_state,
        Country_Region AS country,
        Lat AS lat,
        Long_ AS lon,
        COALESCE(Confirmed, 0) AS confirmed,
        COALESCE(Deaths, 0) AS deaths,
        COALESCE(Recovered, 0) AS recovered,
        COALESCE(Active, 0) AS active,
        FIPS,
        Incident_Rate AS incident_rate,
        Total_Test_Results AS total_test_results,
        People_Hospitalized AS people_hospitalized,
        Case_Fatality_Ratio AS case_fatality_ratio,
        UID,
        ISO3,
        Testing_Rate AS testing_rate,
        Hospitalization_Rate AS hospitalization_rate,
        People_Tested AS people_tested,
        Mortality_Rate AS mortality_rate
    FROM read_csv_auto('{csv_path_us}', union_by_name=true, filename=true);
    """)
    return conn

try:
    with st.spinner("⚡ Connecting to DuckDB & indexing CSSE daily reports..."):
        conn = get_db_connection()
except Exception as e:
    st.error(f"Error loading DuckDB dataset: {e}")
    st.stop()

# Cache date boundary queries
@st.cache_data
def get_date_bounds():
    min_w, max_w = conn.execute("SELECT MIN(report_date), MAX(report_date) FROM clean_world_reports").fetchone()
    min_u, max_u = conn.execute("SELECT MIN(report_date), MAX(report_date) FROM clean_us_reports").fetchone()
    min_date = min(min_w, min_u) if min_w and min_u else (min_w or min_u)
    max_date = max(max_w, max_u) if max_w and max_u else (max_w or max_u)
    return min_date, max_date

min_date, max_date = get_date_bounds()

# Cache Country and State Lists
@st.cache_data
def get_filter_lists():
    countries = [r[0] for r in conn.execute("SELECT DISTINCT country FROM clean_world_reports WHERE country IS NOT NULL ORDER BY country").fetchall()]
    us_states = [r[0] for r in conn.execute("SELECT DISTINCT province_state FROM clean_us_reports WHERE province_state IS NOT NULL ORDER BY province_state").fetchall()]
    return countries, us_states

country_list, us_state_list = get_filter_lists()

# 3. Sidebar Filters
st.sidebar.markdown("### 🛠️ Analytics Settings")

metric_options = {
    "Confirmed Cases": "confirmed",
    "Deaths": "deaths",
    "Recovered": "recovered",
    "Active Cases": "active",
    "Incident Rate (per 100k)": "incident_rate",
    "Case Fatality Ratio (%)": "case_fatality_ratio"
}

selected_metric_label = st.sidebar.selectbox("Select Primary Metric", list(metric_options.keys()))
selected_metric_col = metric_options[selected_metric_label]

st.sidebar.markdown("### 📅 Date Range Filter")
date_range = st.sidebar.date_input(
    "Date Period",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# Metadata info in sidebar
st.sidebar.markdown(f"""
---
**Dataset Information**
* **Source**: Johns Hopkins University CSSE
* **Global Date Range**: `{min_date}` to `{max_date}`
* **Engine**: DuckDB + Streamlit + Plotly
""")

# 4. Main Dashboard Header
st.markdown('<div class="title-gradient">COVID-19 CSSE Intelligence Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Global & US Daily Reports Data, Trend Analysis, and Multi-Region Comparisons</div>', unsafe_allow_html=True)

# 5. Main Tabs
tab_global, tab_us, tab_compare, tab_explorer = st.tabs([
    "🌎 Global Analytics",
    "🇺🇸 US State Analytics",
    "📊 Multi-Region Comparison",
    "🔍 Data Search & Explorer"
])

# ----------------- TAB 1: GLOBAL ANALYTICS -----------------
with tab_global:
    st.markdown("### 🌎 Global COVID-19 Summary & Trends")
    
    @st.cache_data
    def get_global_kpis(dt_end):
        res = conn.execute("""
            SELECT 
                SUM(confirmed) AS total_confirmed,
                SUM(deaths) AS total_deaths,
                SUM(recovered) AS total_recovered,
                SUM(active) AS total_active,
                AVG(case_fatality_ratio) AS avg_cfr
            FROM clean_world_reports
            WHERE report_date = ?
        """, [dt_end]).fetchone()
        return res

    latest_kpis = get_global_kpis(end_date)
    if latest_kpis and latest_kpis[0] is not None:
        c_conf, c_death, c_rec, c_active, c_cfr = latest_kpis
        cfr_val = (c_death / c_conf * 100) if (c_conf and c_conf > 0) else (c_cfr or 0)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-label">Global Confirmed Cases</div>
                <div class="metric-value">{format_num(c_conf)}</div>
                <div class="metric-sub">As of {end_date}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Global Deaths</div>
                <div class="metric-value">{format_num(c_death)}</div>
                <div class="metric-sub">As of {end_date}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Global Recovered</div>
                <div class="metric-value">{format_num(c_rec)}</div>
                <div class="metric-sub">As of {end_date}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Global Active Cases</div>
                <div class="metric-value">{format_num(c_active)}</div>
                <div class="metric-sub">As of {end_date}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Case Fatality Ratio</div>
                <div class="metric-value">{cfr_val:.2f}%</div>
                <div class="metric-sub">Overall Mortality Rate</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No global summary records available for the selected date.")

    # Global Time Series Trend
    st.markdown(f"#### 📈 Global {selected_metric_label} Trend ({start_date} to {end_date})")
    
    @st.cache_data
    def get_global_trend(s_date, e_date, metric_col):
        if metric_col in ["incident_rate", "case_fatality_ratio"]:
            agg_type = "AVG"
        else:
            agg_type = "SUM"
            
        df = conn.execute(f"""
            SELECT report_date, {agg_type}({metric_col}) as val
            FROM clean_world_reports
            WHERE report_date BETWEEN ? AND ?
            GROUP BY report_date
            ORDER BY report_date ASC
        """, [s_date, e_date]).fetchdf()
        return df

    g_trend_df = get_global_trend(start_date, end_date, selected_metric_col)
    
    if len(g_trend_df) > 0:
        fig_gtrend = px.line(
            g_trend_df,
            x="report_date",
            y="val",
            labels={"report_date": "Date", "val": selected_metric_label}
        )
        fig_gtrend.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20),
            hovermode="x unified",
            height=380
        )
        fig_gtrend.update_traces(line_color="#3b82f6", line_width=3)
        st.plotly_chart(fig_gtrend, use_container_width=True)
    else:
        st.info("No trend data available for selected date range.")

    st.markdown("---")

    # Global Map & Top Countries Split
    col_map, col_top = st.columns([3, 2])
    
    @st.cache_data
    def get_country_map_data(dt_end, metric_col):
        if metric_col in ["incident_rate", "case_fatality_ratio"]:
            agg_type = "AVG"
        else:
            agg_type = "SUM"
            
        df = conn.execute(f"""
            SELECT country, {agg_type}({metric_col}) as metric_val, AVG(lat) as lat, AVG(lon) as lon
            FROM clean_world_reports
            WHERE report_date = ? AND country IS NOT NULL
            GROUP BY country
            HAVING metric_val IS NOT NULL AND metric_val > 0
            ORDER BY metric_val DESC
        """, [dt_end]).fetchdf()
        return df

    map_df = get_country_map_data(end_date, selected_metric_col)
    
    with col_map:
        st.markdown(f"#### 🗺️ Country Distribution Map ({end_date})")
        if len(map_df) > 0:
            fig_map = px.choropleth(
                map_df,
                locationmode="country names",
                locations="country",
                color="metric_val",
                hover_name="country",
                hover_data={"metric_val": ":,.2f"},
                color_continuous_scale=px.colors.sequential.Plasma,
                labels={"metric_val": selected_metric_label}
            )
            fig_map.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                height=420,
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    coastlinecolor="rgba(255,255,255,0.15)",
                    projection_type="natural earth",
                    bgcolor="rgba(0,0,0,0)",
                    landcolor="#1e293b",
                    lakecolor="#0f172a"
                )
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No geographic distribution data available.")

    with col_top:
        st.markdown(f"#### 🏆 Top 15 Countries by {selected_metric_label}")
        if len(map_df) > 0:
            top15_df = map_df.head(15).sort_values("metric_val", ascending=True)
            fig_top = px.bar(
                top15_df,
                x="metric_val",
                y="country",
                orientation="h",
                color="metric_val",
                color_continuous_scale=px.colors.sequential.Viridis,
                labels={"metric_val": selected_metric_label, "country": "Country"}
            )
            fig_top.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                height=420,
                coloraxis_showscale=False,
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info("No country leaderboard data available.")

    # Country Specific Deep Dive
    st.markdown("---")
    st.markdown("#### 📍 Single Country Deep-Dive")
    selected_country = st.selectbox("Select Country", country_list, index=country_list.index("US") if "US" in country_list else 0)
    
    @st.cache_data
    def get_country_detail(country_name, s_date, e_date):
        df = conn.execute("""
            SELECT report_date, SUM(confirmed) as confirmed, SUM(deaths) as deaths, SUM(recovered) as recovered, SUM(active) as active, AVG(case_fatality_ratio) as case_fatality_ratio
            FROM clean_world_reports
            WHERE country = ? AND report_date BETWEEN ? AND ?
            GROUP BY report_date
            ORDER BY report_date ASC
        """, [country_name, s_date, e_date]).fetchdf()
        return df

    cdetail_df = get_country_detail(selected_country, start_date, end_date)
    
    if len(cdetail_df) > 0:
        fig_cdetail = px.line(
            cdetail_df,
            x="report_date",
            y=["confirmed", "deaths", "recovered", "active"],
            labels={"report_date": "Date", "value": "Count", "variable": "Metric"}
        )
        fig_cdetail.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=10, b=20),
            hovermode="x unified",
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cdetail, use_container_width=True)
    else:
        st.info(f"No detailed records found for {selected_country}.")

# ----------------- TAB 2: US STATE ANALYTICS -----------------
with tab_us:
    st.markdown("### 🇺🇸 US State COVID-19 Analytics")
    
    # Specific US metrics dictionary
    us_metric_options = {
        "Confirmed Cases": "confirmed",
        "Deaths": "deaths",
        "Total Test Results": "total_test_results",
        "People Hospitalized": "people_hospitalized",
        "Incident Rate (per 100k)": "incident_rate",
        "Testing Rate": "testing_rate",
        "Hospitalization Rate": "hospitalization_rate",
        "Case Fatality Ratio (%)": "case_fatality_ratio",
        "Mortality Rate": "mortality_rate"
    }
    
    us_sel_label = st.selectbox("Select US Metric View", list(us_metric_options.keys()))
    us_sel_col = us_metric_options[us_sel_label]
    
    @st.cache_data
    def get_us_kpis(dt_end):
        res = conn.execute("""
            SELECT 
                SUM(confirmed) AS total_confirmed,
                SUM(deaths) AS total_deaths,
                SUM(total_test_results) AS total_tests,
                SUM(people_hospitalized) AS total_hosp,
                AVG(case_fatality_ratio) AS avg_cfr
            FROM clean_us_reports
            WHERE report_date = ?
        """, [dt_end]).fetchone()
        return res

    us_kpis = get_us_kpis(end_date)
    if us_kpis and us_kpis[0] is not None:
        u_conf, u_death, u_tests, u_hosp, u_cfr = us_kpis
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-label">US Total Confirmed</div>
                <div class="metric-value">{format_num(u_conf)}</div>
                <div class="metric-sub">As of {end_date}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">US Total Deaths</div>
                <div class="metric-value">{format_num(u_death)}</div>
                <div class="metric-sub">As of {end_date}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">US Total Tests</div>
                <div class="metric-value">{format_num(u_tests)}</div>
                <div class="metric-sub">As of {end_date}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">People Hospitalized</div>
                <div class="metric-value">{format_num(u_hosp)}</div>
                <div class="metric-sub">Cumulative / Recorded</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Case Fatality Ratio</div>
                <div class="metric-value">{u_cfr:.2f}%</div>
                <div class="metric-sub">US Average CFR</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # US State Map & Leaderboard
    col_us_map, col_us_bar = st.columns([3, 2])
    
    @st.cache_data
    def get_us_state_summary(dt_end, metric_col):
        if metric_col in ["incident_rate", "testing_rate", "hospitalization_rate", "case_fatality_ratio", "mortality_rate"]:
            agg_type = "AVG"
        else:
            agg_type = "SUM"
            
        df = conn.execute(f"""
            SELECT province_state, ISO3, {agg_type}({metric_col}) as metric_val
            FROM clean_us_reports
            WHERE report_date = ? AND province_state IS NOT NULL
            GROUP BY province_state, ISO3
            HAVING metric_val IS NOT NULL AND metric_val > 0
            ORDER BY metric_val DESC
        """, [dt_end]).fetchdf()
        return df

    us_state_df = get_us_state_summary(end_date, us_sel_col)
    
    with col_us_map:
        st.markdown(f"#### 🗺️ US State Map - {us_sel_label}")
        if len(us_state_df) > 0:
            fig_us_map = px.choropleth(
                us_state_df,
                locations="province_state",
                locationmode="USA-states",
                color="metric_val",
                scope="usa",
                hover_name="province_state",
                hover_data={"metric_val": ":,.2f"},
                color_continuous_scale=px.colors.sequential.Tealgrn,
                labels={"metric_val": us_sel_label}
            )
            fig_us_map.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                height=400
            )
            st.plotly_chart(fig_us_map, use_container_width=True)
        else:
            st.info("No state map data available for selected metric.")

    with col_us_bar:
        st.markdown(f"#### 📊 Top US States by {us_sel_label}")
        if len(us_state_df) > 0:
            top_us_df = us_state_df.head(15).sort_values("metric_val", ascending=True)
            fig_us_bar = px.bar(
                top_us_df,
                x="metric_val",
                y="province_state",
                orientation="h",
                color="metric_val",
                color_continuous_scale=px.colors.sequential.Electric,
                labels={"metric_val": us_sel_label, "province_state": "State"}
            )
            fig_us_bar.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                height=400,
                coloraxis_showscale=False,
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_us_bar, use_container_width=True)
        else:
            st.info("No US state bar chart data available.")

    # State Level Deep Dive
    st.markdown("---")
    st.markdown("#### 🏛️ Single State Historical Deep-Dive")
    selected_state = st.selectbox("Choose US State", us_state_list, index=us_state_list.index("California") if "California" in us_state_list else 0)
    
    @st.cache_data
    def get_state_history(state_name, s_date, e_date):
        df = conn.execute("""
            SELECT report_date, confirmed, deaths, total_test_results, people_hospitalized, incident_rate, case_fatality_ratio
            FROM clean_us_reports
            WHERE province_state = ? AND report_date BETWEEN ? AND ?
            ORDER BY report_date ASC
        """, [state_name, s_date, e_date]).fetchdf()
        return df

    state_hist_df = get_state_history(selected_state, start_date, end_date)
    
    if len(state_hist_df) > 0:
        fig_shist = px.line(
            state_hist_df,
            x="report_date",
            y=["confirmed", "deaths", "total_test_results", "people_hospitalized"],
            labels={"report_date": "Date", "value": "Count", "variable": "Metric"}
        )
        fig_shist.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=10, b=20),
            hovermode="x unified",
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_shist, use_container_width=True)
    else:
        st.info(f"No historical records found for state: {selected_state}.")

# ----------------- TAB 3: MULTI-REGION COMPARISON -----------------
with tab_compare:
    st.markdown("### 📊 Multi-Region & Multi-State Comparison Tool")
    st.markdown("Overlay and compare multiple countries or US states directly on the same timeline.")
    
    compare_mode = st.radio("Select Comparison Mode", ["🌍 Compare Countries", "🇺🇸 Compare US States"], horizontal=True)
    
    if compare_mode == "🌍 Compare Countries":
        default_c = ["US", "Brazil", "India", "United Kingdom", "France"]
        valid_c = [c for c in default_c if c in country_list]
        selected_compare_c = st.multiselect("Select Countries to Overlay", country_list, default=valid_c[:3] if valid_c else country_list[:2])
        
        if selected_compare_c:
            @st.cache_data
            def get_multi_country_comp(c_list, s_date, e_date, metric_col):
                if metric_col in ["incident_rate", "case_fatality_ratio"]:
                    agg_type = "AVG"
                else:
                    agg_type = "SUM"
                    
                placeholders = ",".join(["?"] * len(c_list))
                df = conn.execute(f"""
                    SELECT country, report_date, {agg_type}({metric_col}) as val
                    FROM clean_world_reports
                    WHERE country IN ({placeholders}) AND report_date BETWEEN ? AND ?
                    GROUP BY country, report_date
                    ORDER BY report_date ASC
                """, c_list + [s_date, e_date]).fetchdf()
                return df

            comp_df = get_multi_country_comp(selected_compare_c, start_date, end_date, selected_metric_col)
            
            if len(comp_df) > 0:
                fig_ccomp = px.line(
                    comp_df,
                    x="report_date",
                    y="val",
                    color="country",
                    labels={"report_date": "Date", "val": selected_metric_label, "country": "Country"}
                )
                fig_ccomp.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=10, b=20),
                    hovermode="x unified",
                    height=480
                )
                st.plotly_chart(fig_ccomp, use_container_width=True)
            else:
                st.info("No comparative records found for selected countries.")
        else:
            st.info("Please select one or more countries to display the comparison plot.")

    else: # US States Comparison
        default_s = ["California", "Texas", "Florida", "New York", "Illinois"]
        valid_s = [s for s in default_s if s in us_state_list]
        selected_compare_s = st.multiselect("Select US States to Overlay", us_state_list, default=valid_s[:3] if valid_s else us_state_list[:2])
        
        if selected_compare_s:
            @st.cache_data
            def get_multi_state_comp(s_list, s_date, e_date, metric_col):
                if metric_col in ["incident_rate", "testing_rate", "hospitalization_rate", "case_fatality_ratio", "mortality_rate"]:
                    agg_type = "AVG"
                else:
                    agg_type = "SUM"
                    
                placeholders = ",".join(["?"] * len(s_list))
                df = conn.execute(f"""
                    SELECT province_state, report_date, {agg_type}({metric_col}) as val
                    FROM clean_us_reports
                    WHERE province_state IN ({placeholders}) AND report_date BETWEEN ? AND ?
                    GROUP BY province_state, report_date
                    ORDER BY report_date ASC
                """, s_list + [s_date, e_date]).fetchdf()
                return df

            s_comp_df = get_multi_state_comp(selected_compare_s, start_date, end_date, us_sel_col)
            
            if len(s_comp_df) > 0:
                fig_scomp = px.line(
                    s_comp_df,
                    x="report_date",
                    y="val",
                    color="province_state",
                    labels={"report_date": "Date", "val": us_sel_label, "province_state": "State"}
                )
                fig_scomp.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=10, b=20),
                    hovermode="x unified",
                    height=480
                )
                st.plotly_chart(fig_scomp, use_container_width=True)
            else:
                st.info("No comparative records found for selected US states.")
        else:
            st.info("Please select one or more US states to display the comparison plot.")

# ----------------- TAB 4: DATA SEARCH & EXPLORER -----------------
with tab_explorer:
    st.markdown("### 🔍 Interactive Data Explorer & Export")
    st.markdown("Query raw records from CSSE Global or US datasets and download filtered subsets.")
    
    explorer_dataset = st.radio("Select Target Dataset", ["Global Daily Reports", "US Daily Reports"], horizontal=True)
    
    col_exp1, col_exp2 = st.columns(2)
    
    if explorer_dataset == "Global Daily Reports":
        with col_exp1:
            exp_country = st.multiselect("Filter by Country", country_list)
        with col_exp2:
            exp_dates = st.date_input("Filter Date Period", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="exp_g_dates")
            
        if isinstance(exp_dates, tuple) and len(exp_dates) == 2:
            ex_start, ex_end = exp_dates
        else:
            ex_start, ex_end = min_date, max_date

        @st.cache_data
        def query_global_explorer(countries, s_dt, e_dt):
            query = "SELECT report_date, country, province_state, confirmed, deaths, recovered, active, incident_rate, case_fatality_ratio FROM clean_world_reports WHERE report_date BETWEEN ? AND ?"
            params = [s_dt, e_dt]
            if countries:
                placeholders = ",".join(["?"] * len(countries))
                query += f" AND country IN ({placeholders})"
                params.extend(countries)
            query += " ORDER BY report_date DESC, confirmed DESC LIMIT 2000"
            return conn.execute(query, params).fetchdf()

        exp_result_df = query_global_explorer(exp_country, ex_start, ex_end)
        
    else: # US Explorer
        with col_exp1:
            exp_state = st.multiselect("Filter by US State", us_state_list)
        with col_exp2:
            exp_dates = st.date_input("Filter Date Period", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="exp_u_dates")

        if isinstance(exp_dates, tuple) and len(exp_dates) == 2:
            ex_start, ex_end = exp_dates
        else:
            ex_start, ex_end = min_date, max_date

        @st.cache_data
        def query_us_explorer(states, s_dt, e_dt):
            query = "SELECT report_date, province_state, confirmed, deaths, total_test_results, people_hospitalized, incident_rate, testing_rate, hospitalization_rate, case_fatality_ratio FROM clean_us_reports WHERE report_date BETWEEN ? AND ?"
            params = [s_dt, e_dt]
            if states:
                placeholders = ",".join(["?"] * len(states))
                query += f" AND province_state IN ({placeholders})"
                params.extend(states)
            query += " ORDER BY report_date DESC, confirmed DESC LIMIT 2000"
            return conn.execute(query, params).fetchdf()

        exp_result_df = query_us_explorer(exp_state, ex_start, ex_end)

    st.markdown(f"Found **{len(exp_result_df):,}** records (showing top 2,000 matches).")
    
    if len(exp_result_df) > 0:
        st.dataframe(exp_result_df, use_container_width=True)
        
        csv_data = exp_result_df.to_csv(index=False)
        st.download_button(
            label="💾 Download Export CSV",
            data=csv_data,
            file_name=f"csse_covid_export_{explorer_dataset.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No matching records found for active filters.")