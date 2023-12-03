import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from pycountry import countries as ctr
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================
# Set up data
# ==========================

# worldgeo_path = "streamlit_datasets/worldgeo.json"
monthly_sales_path = "streamlit_datasets/country_monthly_sales.csv"
product_sales_path = "streamlit_datasets/country_product_sales.csv"
co2_sale_path = "streamlit_datasets/estimated_co2_per_sale.csv"
flow_path = "streamlit_datasets/flow_emissions_data.csv"

# worldgeo_json = json.load(open(worldgeo_path, "r", encoding="utf-8"))
monthly_sales_df = pd.read_csv(monthly_sales_path)
product_sales_df = pd.read_csv(product_sales_path)
co2_sale_df = pd.read_csv(co2_sale_path)
flow_df = pd.read_csv(flow_path)

YEARS = [2020, 2021, 2022, 2023]
METRICS = ["Sales", "Emissions"]
CODES = ["Country", "Site"]

SCENARIOS = ["Business as usual", "-5% emissions", "-10% emissions", "-20% emissions"]


# ==========================
# Utils functions
# ==========================
def get_yearly_sales_map(year, metric, code_type):
    if metric == "Sales":
        df = monthly_sales_df.copy()
        df = df[df["Year"] == year]
        fig = px.choropleth(
            df,
            locations="iso",
            color="Yearly Total",
            color_continuous_scale=px.colors.sequential.speed,
            # mapbox_style="carto-positron",
            # zoom=1,
            # center={"lat": 0, "lon": 0},
            projection="natural earth",
        )
    else:
        df = flow_df.copy()
        df2 = monthly_sales_df.copy()
        ratio = (
            df2[df2["Year"] == year]["Yearly Total"].sum() / df2["Yearly Total"].sum()
        )
        if code_type == "Country":
            df = df.groupby("Country Code").agg(
                {"Estimated CO2 Emissions (kg)": "sum", "Country Code": "max"}
            )
            df["Estimated CO2 Emissions (kg)"] = df[
                "Estimated CO2 Emissions (kg)"
            ] * np.full(df.shape[0], ratio)
            df = df.rename(columns={"Country Code": "iso"})
        else:
            df = df.groupby("Site Country Code").agg(
                {
                    "Estimated CO2 Emissions (kg)": "sum",
                    "Site Country Code": "max",
                },
            )
            df["Estimated CO2 Emissions (kg)"] = df[
                "Estimated CO2 Emissions (kg)"
            ] * np.full(df.shape[0], ratio)
            df = df[df["Site Country Code"] != "OO"].rename(
                columns={"Site Country Code": "iso"}
            )
        df["iso"] = df["iso"].apply(lambda x: ctr.get(alpha_2=x).alpha_3)
        print(df.head())
        fig = px.choropleth(
            df,
            locations="iso",
            color="Estimated CO2 Emissions (kg)",
            color_continuous_scale=px.colors.sequential.speed,
            # mapbox_style="carto-positron",
            # zoom=1,
            # center={"lat": 0, "lon": 0},
            projection="natural earth",
        )
    return fig


def get_monthly_sales_bar_chart(year, country):
    df = monthly_sales_df.copy()
    filtered_data = df[(df["Year"] == year) & (df["Country Name"] == country)]

    monthly_sales = filtered_data.iloc[
        0, 3:15
    ]

    plt.figure(figsize=(10, 6))
    plt.title(f"Monthly Sales for {country} in {year}")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.xticks(
        range(12),
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
    )
    fig, ax = plt.subplots()
    ax.bar(monthly_sales.index, monthly_sales.values, color="skyblue")
    return fig


def get_total_co2_year_emissions(year):
    df = co2_sale_df.copy()
    df2 = monthly_sales_df.copy()
    emissions = df["Estimated CO2 Emissions (kg)"].sum() * (
        df2[df2["Year"] == year]["Yearly Total"].sum() / df2["Yearly Total"].sum()
    )
    return emissions


def get_top_customers_bar_chart(year):
    df = monthly_sales_df.copy()
    df = df[df["Year"] == year][
        ["Country", "Yearly Total", "Country Name"]
    ].sort_values(by="Yearly Total", ascending=False)
    print(df.head())
    fig = px.bar(
        data_frame=df,
        x="Country",
        y="Yearly Total",
        hover_data=["Country Name"],
        color="Country",
        title=f"Sales per country in {year}",
        color_discrete_sequence=["blue", "red"],
    )
    return fig


def get_flow_emissions_heatmap():
    df = flow_df.copy()

    pivot_data = df.pivot(
        index="Country Code",
        columns="Site Country Code",
        values="Estimated CO2 Emissions (kg)"
    )

    pivot_data_non_zero = pivot_data.replace(0, np.nan)

    order_of_magnitude = np.floor(np.log10(pivot_data_non_zero))

    plt.figure(figsize=(12, 8))
    heatmap_updated = sns.heatmap(
        order_of_magnitude,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        cbar_kws={"label": "Order of Magnitude (log10)"},
    )
    plt.title(
        "Heatmap of CO2 Emissions (Order of Magnitude) Between Countries and Sites"
    )
    plt.ylabel("Country")
    plt.xlabel("Site Country")
    heatmap_updated.set_facecolor("lightgrey")
    fig = heatmap_updated.get_figure()
    return fig

def get_emissions_scenario_numbers(scenario):
    baseline = get_total_co2_year_emissions(2023)
    if scenario == "Business as usual":
        return (baseline, baseline)
    elif scenario == "-5% emissions":
        return (baseline, baseline * 0.95)
    elif scenario == "-10% emissions":
        return (baseline, baseline * 0.9)
    elif scenario == "-20% emissions":
        return (baseline, baseline * 0.8)


def get_emissions_scenarios_bar_chart(scenario):
    df = flow_df.copy()
    df2 = monthly_sales_df.copy()
    ratio = (
        df2[df2["Year"] == 2020]["Yearly Total"].sum() / df2["Yearly Total"].sum()
    )
    df = df.groupby("Country Code").agg(
        {"Estimated CO2 Emissions (kg)": "sum", "Country Code": "max"}
    )
    df["Estimated CO2 Emissions (kg)"] = df[
        "Estimated CO2 Emissions (kg)"
    ] * np.full(df.shape[0], ratio)
    df = df.rename(columns={"Country Code": "iso"})
    df["iso"] = df["iso"].apply(lambda x: ctr.get(alpha_2=x).alpha_3)
    df["Estimated CO2 Emissions (kg)"] = (
        df["Estimated CO2 Emissions (kg)"]
        * np.full(df.shape[0], get_scenario_coefficients(scenario))
    )
    # Generate a bar chart
    fig = px.bar(
        data_frame=df,
        x="iso",
        y="Estimated CO2 Emissions (kg)",
        hover_data=["iso"],
        color="iso",
        title=f"Estimated CO2 Emissions (kg) per country related to your shipping in 2020",
        color_discrete_sequence=["blue", "red"],
    )
    return fig


def format_big_num(number):
    return f"{number:,.0f}"


def get_scenario_coefficients(scenario):
    if scenario == "Business as usual":
        return 1
    elif scenario == "-5% emissions":
        return 0.95
    elif scenario == "-10% emissions":
        return 0.9
    elif scenario == "-20% emissions":
        return 0.8

formatted_number_no_decimals = format_big_num(1000000)


# ==========================
# User interface
# ==========================

with st.sidebar:
    st.title("Century 27")
    st.image("logo-hi-paris.png")
    st.subheader("Lucas")
    st.subheader("Martin")
    st.subheader("Noureddine")
    st.subheader("Salim")
    st.subheader("Théo")
    st.subheader("Tim")

(tab_worldview, tab_charts, tab_scenarios) = st.tabs(
    [
        "Worldview",
        "Charts",
        "Scenarios",
    ]
)

# ==========================
# Map
# ==========================

tab_worldview.header("Worldview")
tab_worldview.subheader("Get a global overview of your supply chain operations")
year_selector_1 = tab_worldview.selectbox("Year data", YEARS, key="year_selector_1")
tab_worldview.subheader(
    f"You have emitted {get_total_co2_year_emissions(year_selector_1):,.0f} kg of CO2 in {year_selector_1}"
)
code_type_selector_1 = tab_worldview.selectbox(
    "Country or site", CODES, key="code_type_selector_1"
)
metric_selector_1 = tab_worldview.selectbox("Metric", METRICS, key="metric_selector_1")

tab_worldview.plotly_chart(
    get_yearly_sales_map(year_selector_1, metric_selector_1, code_type_selector_1)
)

# figure_1, figure_2 = get_employed_bias_pie_figure(criteria_selector_1)
# tab_analysis_employment.plotly_chart(figure_1)
# tab_analysis_employment.plotly_chart(figure_2)

# ==========================
# Charts
# ==========================

tab_charts.header("Charts")
tab_charts.subheader("Get a detailed overview of your supply chain operations")

year_selector_2 = tab_charts.selectbox("Year data", YEARS, key="year_selector_2")

(
    tab_charts_sales_recap,
    tab_charts_monthly_sales,
    tab_charts_heatmap,
) = tab_charts.tabs(["Sales recap", "Monthly sales", "Heatmap"])

tab_charts_sales_recap.plotly_chart(get_top_customers_bar_chart(year_selector_2))

country_selector_2 = tab_charts_monthly_sales.selectbox(
    "Country", list(monthly_sales_df["Country Name"].unique()), key="country_selector_2"
)
tab_charts_monthly_sales.plotly_chart(
    get_monthly_sales_bar_chart(year_selector_2, country_selector_2)
)

tab_charts_heatmap.pyplot(get_flow_emissions_heatmap())

# ==========================
# Scenarios
# ==========================

tab_scenarios.header("Scenarios")
tab_scenarios.subheader("Get an overview of the future and act on it")

scenario_selector_1 = tab_scenarios.selectbox(
    "Scenario", SCENARIOS, key="scenario_selector_1"
)

baseline_emissions, scenario_emissions = get_emissions_scenario_numbers(scenario_selector_1)

if scenario_selector_1 == "Business as usual":
    tab_scenarios.markdown(
        f"Your emissions will be **{format_big_num(baseline_emissions)} kg** of CO2 in 2023, sales will be **{format_big_num(monthly_sales_df[monthly_sales_df['Year'] == 2023]['Yearly Total'].sum())}** units"
    )
else:
    tab_scenarios.markdown(f"Going from *'Business as usual'* to *'{scenario_selector_1}'* will reduce your emissions by **{format_big_num(baseline_emissions - scenario_emissions)} kg** of CO2 in 2023, sales would be **{format_big_num(monthly_sales_df[monthly_sales_df['Year'] == 2023]['Yearly Total'].sum() * get_scenario_coefficients(scenario_selector_1))}** units")

tab_scenarios.plotly_chart(
    get_emissions_scenarios_bar_chart(scenario_selector_1)
)

# # ==========================
# # Logistic Regression
# # ==========================

# tab_logistic_regression.header("Logistic Regression")
# list_col = tab_logistic_regression.multiselect(
#     "Select variable for Logistic Regress: ",
#     VAL_COLS + TO_DUMMIES,
#     default=VAL_COLS[1:] + TO_DUMMIES,
# )
# result_df, score, X, delta_prob = get_data_log_regression(parameters=list_col)
# tab_logistic_regression.subheader(f"The score is: {round(score * 100, 2)}%")
# tab_logistic_regression.table(result_df)

# # ==========================
# # Fairness test
# # ==========================

# tab_fairness_test.header("Models on biased dataset performance:")

# tab_fairness_test.subheader("Decision Tree performance")
# tab_fairness_test.table(exp1.model_performance().result)

# tab_fairness_test.subheader("Random Forest performance")
# tab_fairness_test.table(exp2.model_performance().result)

# tab_fairness_test.subheader("Gradient Boosting performance")
# tab_fairness_test.table(exp4.model_performance().result)

# tab_fairness_test.header("Fairness check")

# criteria_selector_3 = tab_fairness_test.selectbox(
#     "Which criteria to check fairness on ?",
#     ["Age", "Gender", "MentalHealth", "Accessibility"],
# )

# criteria_selector_4 = tab_fairness_test.selectbox(
#     'Which value to be considered as "privileged" ?', set(df[criteria_selector_3])
# )

# plot = get_fairness_check(criteria_selector_3, criteria_selector_4)

# (
#     t5_fairness_check,
#     t5_metric_scores,
#     t5_stacked,
#     t5_radar,
#     t5_performance_and_fairness,
#     t5_heatmap,
# ) = tab_fairness_test.tabs(
#     [
#         "Fairness Check",
#         "Metric Scores",
#         "Cumulated parity loss",
#         "Radar",
#         "Performance And Fairness",
#         "Heatmap",
#     ]
# )

# t5_fairness_check.plotly_chart(
#     plot("fairness_check"), theme=None, use_container_width=True
# )
# t5_metric_scores.plotly_chart(
#     plot("metric_scores"), theme=None, use_container_width=True
# )
# t5_stacked.plotly_chart(plot("stacked"), theme=None, use_container_width=True)
# t5_radar.plotly_chart(plot("radar"), theme=None, use_container_width=True)
# t5_performance_and_fairness.plotly_chart(
#     plot("performance_and_fairness"), theme=None, use_container_width=True
# )
# t5_heatmap.plotly_chart(plot("heatmap"), theme=None, use_container_width=True)

# # ==========================
# # Bias mitigation
# # ==========================

# tab_bias_mitigation.header("Bias mitigation with Dalex")

# model_selector = tab_bias_mitigation.selectbox(
#     "Which model should have its biases mitigated ?",
#     ["Random Forest", "Gradient Boosting"],
#     key="bias6_model_selectbox",
# )

# criteria_selector_5 = tab_bias_mitigation.selectbox(
#     "Which criteria to check fairness on ?", ["Gender"], key="bias6_1_selectbox"
# )

# criteria_selector_6 = tab_bias_mitigation.selectbox(
#     'Which value to be considered as "privileged" ?', ["Man"], key="bias6_2_selectbox"
# )

# plot = get_fairness_check_after_mitigation(
#     criteria_selector_5, criteria_selector_6, model_selector
# )

# (
#     t6_fairness_check,
#     t6_metric_scores,
#     t6_stacked,
#     t6_radar,
#     t6_performance_and_fairness,
#     t6_heatmap,
# ) = tab_bias_mitigation.tabs(
#     [
#         "Fairness Check",
#         "Metric Scores",
#         "Cumulated parity loss",
#         "Radar",
#         "Performance And Fairness",
#         "Heatmap",
#     ]
# )

# t6_fairness_check.plotly_chart(
#     plot("fairness_check"), theme=None, use_container_width=True
# )
# t6_metric_scores.plotly_chart(
#     plot("metric_scores"), theme=None, use_container_width=True
# )
# t6_stacked.plotly_chart(plot("stacked"), theme=None, use_container_width=True)
# t6_radar.plotly_chart(plot("radar"), theme=None, use_container_width=True)
# t6_performance_and_fairness.plotly_chart(
#     plot("performance_and_fairness"), theme=None, use_container_width=True
# )
# t6_heatmap.plotly_chart(plot("heatmap"), theme=None, use_container_width=True)
