#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_strategy_dashboard.py

An enhanced multi–strategy trade list dashboard with dark UI.

Features:
  • Two file input modes: multi–file upload OR folder path.
  • Sidebar options: decimal precision, strategy selection.
  • 10 individual analysis tabs (Trade List, Performance Summary, Advanced Metrics,
    Entry Size Stats, Aggregated Stats, Extra Info, Trend Group Stats, Drawdown & Recovery,
    Profit Factor Breakdown, Time-of-Day Analysis, Clustering Analysis) with all original
    metrics, charts, commentary plus new enhancements.
  • 6 additional comparison tabs:
      1. Strategy Comparison Summary
      2. Equity Curve Comparison
      3. Advanced Metrics Comparison
      4. Strategy Ranking & Composite Score
      5. Entry Size Stats Comparison Across Strategies
      6. Strategy Judgement Ratio Comparison

Additional Enhancements:
  • New Filters: Date Range for Entry_Date/Exit_Date, Entry_Price & Exit_Price filters.
  • New Charts: Scatter plot (Entry_Price vs Exit_Price), Histogram/Violin for PnL_per_Lot,
    Density/CDF plots, regression plots, heatmaps, area charts, polar charts.
  • Expanded Metrics: Trade Duration Distribution (mean, median, percentiles), trade frequency,
    risk–reward distribution, win/loss counts, VaR, CDaR, skewness, kurtosis, moving averages,
    rolling metrics.
  • Clustering enhancements: Silhouette Score, Elbow Method chart, enhanced radar charts.
  • New comparative tabs for Entry Size Stats & Strategy Judgement Ratio comparisons.

Enhancements in this version:
  • Optional combining of multiple trade sheets into one unified trade list,
    sorted by either Entry_Date or Exit_Date, to view aggregated results as
    if it were a single strategy.
  • Strategy Comparison Summary now includes *all* performance summary metrics
    (consecutive wins, current drawdown, etc.) and offers a filter for every
    numeric column in the comparison table.
  • **Detailed Outperformance Analysis per Lot Size**: In the Entry Size Stats Comparison tab,
    added detailed breakdown tables comparing strategies against a selected benchmark for each lot size.

Author: You
Date: 2025-03-01
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import os, glob
from datetime import datetime
from scipy import stats

# For PCA and clustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Plotly
import plotly.express as px
import plotly.graph_objects as go

# For interactive data grid
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

#####################################
# CUSTOM CSS FOR DARK THEME
#####################################
custom_css = """
<style>
body {
    background-color: #121212;
    color: #e0e0e0;
}
[data-baseweb="base-input"] {
    background-color: #1e1e1e !important;
}
.stButton>button {
    background-color: #2a2a2a;
    color: #e0e0e0;
    border: none;
}
.stSelectbox, .stNumberInput, .stMultiSelect, .stMetric, .stTextInput {
    background-color: #2a2a2a !important;
    color: #e0e0e0 !important;
}
table {
    width: 100%;
    border-collapse: collapse;
    color: #e0e0e0;
}
table th {
    background-color: #1e1e1e;
    color: #e0e0e0;
    padding: 8px;
    border: 1px solid #333;
}
table td {
    background-color: #242424;
    color: #e0e0e0;
    padding: 8px;
    border: 1px solid #333;
}
table tr:nth-child(even) td {
    background-color: #1e1e1e;
}
.ag-root-wrapper {
    background-color: #242424 !important;
    color: #e0e0e0 !important;
}
.metric-container {
    background-color: #1e1e1e;
    padding: 12px;
    border-radius: 5px;
    margin: 5px;
    text-align: center;
}
.metric-title {
    font-size: 14px;
    color: #b0b0b0;
}
.metric-value {
    font-size: 24px;
    font-weight: bold;
    color: #ffffff;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

#####################################
# HELPER FUNCTIONS
#####################################
@st.cache_data(show_spinner=True)
def load_csv_from_file(file_obj):
    try:
        df = pd.read_csv(file_obj)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

def display_aggrid_table(df, key="grid", page_size=500):
    if df.empty:
        st.write("No data available.")
    else:
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_pagination(paginationAutoPageSize=True, paginationPageSize=page_size)
        gridOptions = gb.build()
        AgGrid(df, gridOptions=gridOptions, update_mode=GridUpdateMode.NO_UPDATE, theme="balhamDark", key=key)

def display_dataframe_with_precision(df, precision):
    format_dict = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            format_dict[col] = "{:." + str(precision) + "f}"
    st.dataframe(df.style.format(format_dict))

def compute_cvar(trade_df, confidence=0.95):
    losses = trade_df["PnL_Currency"][trade_df["PnL_Currency"] < 0]
    if losses.empty:
        return np.nan
    var = np.percentile(losses, (1-confidence)*100)
    cvar = losses[losses <= var].mean()
    return cvar

def compute_trade_return_cdf(trade_df):
    returns = np.sort(trade_df["PnL_Currency"].dropna().values)
    cdf = np.arange(1, len(returns)+1) / len(returns)
    return returns, cdf

def compute_var(trade_df, confidence=0.95):
    losses = trade_df["PnL_Currency"][trade_df["PnL_Currency"] < 0]
    if losses.empty:
        return np.nan
    return np.percentile(losses, (1-confidence)*100)

def compute_cdar(trade_df):
    if "Exit_Date" not in trade_df.columns:
        return np.nan
    ts = trade_df.sort_values("Exit_Date").copy()
    ts["Cumulative_PnL"] = ts["PnL_Currency"].cumsum()
    drawdowns = ts["Cumulative_PnL"] - ts["Cumulative_PnL"].cummax()
    dd_active = (drawdowns < 0)
    if dd_active.sum() == 0:
        return 0
    durations = []
    count = 0
    for active in dd_active:
        if active:
            count += 1
        elif count > 0:
            durations.append(count)
            count = 0
    if count > 0:
        durations.append(count)
    return np.mean(durations) if durations else np.nan

def get_skewness_kurtosis(trade_df):
    pnl = trade_df["PnL_Currency"].dropna()
    return pnl.skew(), pnl.kurtosis()

def moving_average(series, window=20):
    return series.rolling(window=window).mean()

def correlation_matrix(df, cols):
    return df[cols].corr()

from sklearn.preprocessing import LabelEncoder
def encode_features(df, features):
    df_encoded = df.copy()
    for f in features:
        if f in df_encoded.columns:
            if not np.issubdtype(df_encoded[f].dtype, np.number):
                le = LabelEncoder()
                df_encoded[f] = le.fit_transform(df_encoded[f].astype(str))
    return df_encoded

def perform_pca(X):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    return components

#####################################
# COMMENTARY FUNCTIONS
#####################################
def interpret_trade_list(df: pd.DataFrame) -> str:
    lines = []
    lines.append("## Trade List Commentary\n")
    n_trades = len(df)
    lines.append(f"- Detected **{n_trades} trades**.\n")
    if "PnL_Currency" in df.columns and n_trades > 2:
        pnl = df["PnL_Currency"].dropna()
        mean_pnl = pnl.mean()
        std_pnl = pnl.std()
        skew_pnl = pnl.skew()
        kurt_pnl = pnl.kurtosis()
        lines.append(f"- Mean PnL: {mean_pnl:.2f}, Std Dev: {std_pnl:.2f}, Skew: {skew_pnl:.2f}, Kurtosis: {kurt_pnl:.2f}.\n")
        t_stat, p_val = stats.ttest_1samp(pnl, 0)
        if p_val < 0.05 and mean_pnl > 0:
            lines.append("  => Statistically above 0 (p<0.05). Positive expectancy.\n")
        else:
            lines.append("  => Not statistically confirmed >0 at 5% level.\n")
        outlier_threshold = mean_pnl + 3 * std_pnl
        outliers = (pnl > outlier_threshold).sum()
        if outliers > 0:
            lines.append(f"  => {outliers} outliers above mean+3std; these big winners might skew the average.\n")
    if "Duration_Hours" in df.columns:
        neg_duration = (df["Duration_Hours"] < 0).sum()
        if neg_duration > 0:
            lines.append(f"- {neg_duration} trades have negative duration, indicating potential data issues.\n")

    if "Entry_Date" in df.columns:
        dates = pd.to_datetime(df["Entry_Date"])
        freq = dates.dt.date.value_counts().mean()
        lines.append(f"- Average trades per day: {freq:.2f}.\n")
    lines.append("\n**Summary**: Trade-level statistics reveal performance trends and potential anomalies.")
    return "\n".join(lines)

def interpret_performance_summary(win_rate: float, pf: float, achieved_rr: float, max_dd: float,
                                  strat_judgment_ratio: float, total_trades: int, total_pnl: float,
                                  avg_pnl_trade: float, max_cons_losses: int, max_cons_wins: int,
                                  pnl_series: pd.Series=None, trade_dates: pd.Series=None) -> str:
    lines = []
    lines.append("## Performance Summary Commentary\n")
    lines.append(f"- Total Trades: {total_trades}, Net PnL: {total_pnl:.2f}, Avg PnL/trade: {avg_pnl_trade:.2f}.\n")
    lines.append("### Win Rate & Trade Frequency\n")
    if win_rate < 40:
        lines.append(f"- Win Rate={win_rate:.1f}% (low); strategy may rely on occasional big winners.\n")
    elif win_rate < 60:
        lines.append(f"- Win Rate={win_rate:.1f}% (moderate).\n")
    else:
        lines.append(f"- Win Rate={win_rate:.1f}% (high); frequent wins observed.\n")
    lines.append("### Profit Factor & Risk-Reward\n")
    if pf < 1.0:
        lines.append(f"- Profit Factor (PF)={pf:.2f}: indicates a losing system.\n")
    elif pf < 1.3:
        lines.append(f"- PF={pf:.2f}: borderline profitability.\n")
    elif pf < 2.0:
        lines.append(f"- PF={pf:.2f}: robust performance.\n")
    else:
        lines.append(f"- PF={pf:.2f}: excellent performance (≥2.0).\n")
    lines.append("\n### Achieved RR\n")
    if not np.isnan(achieved_rr):
        if achieved_rr < 1.0:
            lines.append(f"- Achieved RR={achieved_rr:.2f}: average loser is close to the winner.\n")
        else:
            lines.append(f"- Achieved RR={achieved_rr:.2f}: winners generally outweigh losers.\n")
    else:
        lines.append("- Achieved RR not computed.\n")
    lines.append("\n### Max Drawdown\n")
    if max_dd < 0:
        lines.append(f"- Max Drawdown={max_dd:.2f}: watch for significant dips.\n")
    else:
        lines.append("- Max Drawdown not negative; check data.\n")
    lines.append("\n### Strategy Judgment Ratio\n")
    if not np.isnan(strat_judgment_ratio):
        if strat_judgment_ratio < 0.5:
            lines.append(f"- Strategy Judgment Ratio={strat_judgment_ratio:.2f}: drawdowns may outweigh gains.\n")
        else:
            lines.append(f"- Strategy Judgment Ratio={strat_judgment_ratio:.2f}: good recovery.\n")
    else:
        lines.append("- Strategy Judgment Ratio not available.\n")
    lines.append(f"\n### Consecutive Wins: {max_cons_wins}, Consecutive Losses: {max_cons_losses}\n")
    if pnl_series is not None and trade_dates is not None and len(pnl_series)==len(trade_dates):
        from scipy.stats import linregress
        ordinal_x = pd.to_datetime(trade_dates).apply(lambda d: d.toordinal()).values
        slope, _, r_val, p_val, _ = linregress(ordinal_x, pnl_series.values)
        lines.append("### Chronological Trend in Trade PnL\n")
        lines.append(f"- Slope={slope:.4f}, r={r_val:.2f}, p={p_val:.4f}.\n")
        if p_val < 0.05:
            lines.append("  => Statistically significant trend observed.\n")
    lines.append("\n**Conclusion**: The combination of win rate, risk–reward, and drawdown measures provides a holistic view.")
    return "\n".join(lines)

def interpret_advanced_metrics(sharpe: float, sortino: float, calmar: float, ulcer: float,
                               omega: float, kelly: float, eq_series: pd.Series=None) -> str:
    lines = []
    lines.append("## Advanced Metrics Commentary\n")
    if eq_series is not None and len(eq_series) > 30:
        from scipy.stats import linregress
        x = np.arange(len(eq_series))
        slope, _, r_val, p_val, _ = linregress(x, eq_series.values)
        lines.append("### Equity Trend\n")
        lines.append(f"- Slope={slope:.2f}, r={r_val:.2f}, p={p_val:.4f}.\n")
        if p_val < 0.05:
            lines.append("  => Significant equity trend detected.\n")
        lines.append("\n")
    lines.append("### Sharpe Ratio\n")
    if np.isnan(sharpe):
        lines.append("- Sharpe not computed.\n")
    else:
        lines.append(f"- Sharpe={sharpe:.2f}.\n")
    lines.append("### Sortino Ratio\n")
    if np.isnan(sortino):
        lines.append("- Sortino not computed.\n")
    else:
        lines.append(f"- Sortino={sortino:.2f}.\n")
    lines.append("### Calmar Ratio\n")
    if np.isnan(calmar):
        lines.append("- Calmar not available.\n")
    else:
        lines.append(f"- Calmar={calmar:.2f}.\n")
    lines.append("### Ulcer Index\n")
    if np.isnan(ulcer):
        lines.append("- Ulcer not computed.\n")
    else:
        lines.append(f"- Ulcer={ulcer:.2f}.\n")
    lines.append("### Omega Ratio\n")
    if np.isnan(omega):
        lines.append("- Omega not computed.\n")
    else:
        lines.append(f"- Omega={omega:.2f}.\n")
    lines.append("### Kelly Fraction\n")
    if np.isnan(kelly):
        lines.append("- Kelly fraction not computed.\n")
    else:
        lines.append(f"- Kelly={kelly:.2f}.\n")
    lines.append("\n**Conclusion**: Advanced risk–adjusted metrics shed light on the strategy’s robustness.")
    return "\n".join(lines)

def interpret_entry_size_stats(grouped_df: pd.DataFrame) -> str:
    lines = []
    lines.append("## Entry Size Stats Commentary\n")
    if grouped_df.empty or "Entry_Size" not in grouped_df.columns:
        return "- No entry size data available."
    lines.append(f"- Found {len(grouped_df)} distinct entry sizes.\n")
    for _, row in grouped_df.iterrows():
        es = row["Entry_Size"]
        pf = row.get("Profit_Factor", np.nan)
        lines.append(f"### Entry Size = {es}\n")
        if not np.isnan(pf):
            if pf < 1.0:
                lines.append(f"- PF={pf:.2f}: losing at this size.\n")
            elif pf < 1.3:
                lines.append(f"- PF={pf:.2f}: borderline.\n")
            else:
                lines.append(f"- PF={pf:.2f}: robust performance.\n")
        lines.append("\n")
    lines.append("**Insight**: The impact of trade size on performance is clearly visible.")
    return "\n".join(lines)

def interpret_aggregated_stats(agg_df: pd.DataFrame) -> str:
    lines = []
    lines.append("## Aggregated Stats Commentary\n")
    if agg_df.empty:
        return "- No aggregated data available."
    negatives = agg_df[agg_df["Total_Profit"] < 0]
    if not negatives.empty:
        lines.append(f"- {len(negatives)} periods had negative profit.\n")
        worst = negatives["Total_Profit"].min()
        worst_period = negatives.loc[negatives["Total_Profit"] == worst, "Period"].iloc[0]
        lines.append(f"  => Worst period: {worst_period} with profit {worst:.2f}.\n")
    else:
        lines.append("- All periods showed positive profit.\n")
    lines.append("\n**Insight**: Time-based aggregation reveals seasonality and cyclic performance trends.")
    return "\n".join(lines)

def interpret_extra_info(durations: pd.Series=None, trade_pnl: pd.Series=None) -> str:
    lines = []
    lines.append("## Extra Info Commentary\n")
    if durations is not None and len(durations) > 5:
        mean_dur = durations.mean()
        std_dur = durations.std()
        skew_dur = durations.skew()
        lines.append(f"- Duration: mean={mean_dur:.2f}, std={std_dur:.2f}, skew={skew_dur:.2f}.\n")
    if trade_pnl is not None and len(trade_pnl) > 5:
        stat, pval = stats.normaltest(trade_pnl.dropna())
        lines.append(f"- PnL Normality test p={pval:.4f}.\n")
        if pval < 0.05:
            lines.append("  => PnL deviates from normal distribution.\n")
    if durations is not None and trade_pnl is not None:
        df_temp = pd.DataFrame({"Duration": durations, "PnL": trade_pnl})
        corr = df_temp.corr().iloc[0,1]
        lines.append(f"- Correlation between Duration and PnL: {corr:.2f}.\n")
    lines.append("\n**Summary**: Additional distributional insights help identify outliers and biases.")
    return "\n".join(lines)

def interpret_trend_group_stats(trend_df: pd.DataFrame) -> str:
    lines = []
    lines.append("## Trend Group Stats Commentary\n")
    if trend_df.empty:
        return "- No trend data available."
    if "Total Trend PnL" in trend_df.columns and "Trend Duration (Days)" in trend_df.columns:
        valid_mask = ~trend_df["Total Trend PnL"].isna() & ~trend_df["Trend Duration (Days)"].isna()
        if valid_mask.sum() > 5:
            r, pval = stats.pearsonr(trend_df.loc[valid_mask, "Total Trend PnL"],
                                     trend_df.loc[valid_mask, "Trend Duration (Days)"])
            lines.append(f"- Correlation between Trend PnL and Duration: {r:.2f} (p={pval:.4f}).\n")
            if pval < 0.05 and r > 0:
                lines.append("  => Longer trends tend to yield higher PnL.\n")
    if "Num Trades" in trend_df.columns:
        avg_trades = trend_df["Num Trades"].mean()
        lines.append(f"- Average number of trades per trend: {avg_trades:.2f}.\n")
    lines.append("\n**Insight**: Trend analysis can uncover the benefits of letting winners run.")
    return "\n".join(lines)

def interpret_drawdown_analysis(dd_df: pd.DataFrame) -> str:
    lines = []
    lines.append("## Drawdown & Recovery Commentary\n")
    if dd_df.empty:
        return "- No drawdown data available."
    if "Drawdown" in dd_df.columns:
        worst_dd = dd_df["Drawdown"].min()
        lines.append(f"- Worst drawdown: {worst_dd:.2f}.\n")
    if "Volatility_Zone" in dd_df.columns and "Drawdown_%" in dd_df.columns:
        high_vol_dd = dd_df[dd_df["Volatility_Zone"]=="High"]["Drawdown_%"].mean()
        low_vol_dd  = dd_df[dd_df["Volatility_Zone"]=="Low"]["Drawdown_%"].mean()
        lines.append(f"- Avg Drawdown in High Vol: {high_vol_dd:.2%}, Low Vol: {low_vol_dd:.2%}.\n")
    if "Exit_Date" in dd_df.columns:
        recovery_time = (dd_df["Exit_Date"].iloc[-1] - dd_df["Exit_Date"].iloc[0]).days
        lines.append(f"- Total period for drawdown analysis: {recovery_time} days.\n")
    lines.append("\n**Note**: Drawdown metrics are crucial for understanding risk and recovery potential.")
    return "\n".join(lines)

def interpret_profit_factor_breakdown(pf_dir: pd.DataFrame, pf_vol: pd.DataFrame, pf_exit: pd.DataFrame) -> str:
    lines = []
    lines.append("## Profit Factor Breakdown Commentary\n")
    if pf_dir is not None and not pf_dir.empty:
        lines.append("### By Direction\n")
        for _, row in pf_dir.iterrows():
            lines.append(f"- Direction {row['Direction']}: PF={row['Profit_Factor']:.2f}.\n")
    if pf_vol is not None and not pf_vol.empty:
        lines.append("\n### By Volatility Zone\n")
        for _, row in pf_vol.iterrows():
            lines.append(f"- Zone {row['Volatility_Zone']}: PF={row['Profit_Factor']:.2f}.\n")
    if pf_exit is not None and not pf_exit.empty:
        lines.append("\n### By Exit Method\n")
        for _, row in pf_exit.iterrows():
            lines.append(f"- Exit Method {row['Exit_Method']}: PF={row['Profit_Factor']:.2f}.\n")
    lines.append("\n**Conclusion**: Breaking down PF by various dimensions reveals the underlying performance drivers.")
    return "\n".join(lines)

def interpret_time_of_day_analysis(tod_df: pd.DataFrame, dow_df: pd.DataFrame) -> str:
    lines = []
    lines.append("## Time-of-Day / Day-of-Week Commentary\n")
    lines.append("### Hourly PnL\n")
    if tod_df.empty:
        lines.append("- No time-of-day data available.\n")
    else:
        worst = tod_df.loc[tod_df["sum"].idxmin()]
        lines.append(f"- Worst hour: {worst['Entry_Hour']} with Total PnL={worst['sum']:.2f}.\n")
    lines.append("\n### Day-of-Week PnL\n")
    if dow_df.empty:
        lines.append("- No day-of-week data available.\n")
    else:
        worst2 = dow_df.loc[dow_df["sum"].idxmin()]
        lines.append(f"- Worst day: {worst2['Entry_DayOfWeek']} with Total PnL={worst2['sum']:.2f}.\n")
    lines.append("\n**Conclusion**: Identifying underperforming hours/days can help in optimizing trade timing.")
    return "\n".join(lines)

#####################################
# PERFORMANCE FUNCTIONS
#####################################
@st.cache_data(show_spinner=False)
def compute_group_summary(df, global_decimals=3):
    if df.empty or "PnL_Currency" not in df.columns:
        return pd.DataFrame()
    groups = {"Overall": df}
    if ("Entry_Size" in df.columns) and ("PnL_per_Lot" not in df.columns):
        df["PnL_per_Lot"] = df["PnL_Currency"] / df["Entry_Size"]
    if "Direction" in df.columns:
        groups["Long"] = df[df["Direction"]==1]
        groups["Short"] = df[df["Direction"]==-1]
    if "Volatility_Zone" in df.columns:
        groups["High Vol"] = df[df["Volatility_Zone"]=="High"]
        groups["Low Vol"] = df[df["Volatility_Zone"]=="Low"]
    def metrics_for(tr):
        if tr.empty:
            return {
                "Total Trades": 0,
                "Total PnL": 0,
                "Win Rate (%)": 0,
                "Avg PnL": np.nan,
                "Median PnL": np.nan,
                "Std Dev PnL": np.nan,
                "Gross Profit": 0,
                "Gross Loss": 0,
                "Profit Factor": np.nan,
                "Avg Holding Time (hrs)": np.nan,
                "Avg PnL/Trade/Lot": np.nan,
                "Achieved RR": np.nan
            }
        total_trades = len(tr)
        total_pnl = tr["PnL_Currency"].sum()
        wins = tr[tr["PnL_Currency"] > 0]
        losses = tr[tr["PnL_Currency"] < 0]
        win_rate = (len(wins) / total_trades * 100) if total_trades>0 else 0
        avg_pnl = tr["PnL_Currency"].mean()
        median_pnl = tr["PnL_Currency"].median()
        std_pnl = tr["PnL_Currency"].std()
        gross_profit = wins["PnL_Currency"].sum() if not wins.empty else 0
        gross_loss = losses["PnL_Currency"].sum() if not losses.empty else 0
        profit_factor = gross_profit/abs(gross_loss) if gross_loss<0 else np.nan
        avg_holding = tr["Duration_Hours"].mean() if "Duration_Hours" in tr.columns else np.nan
        avg_pnl_per_lot = tr["PnL_per_Lot"].mean() if "PnL_per_Lot" in tr.columns else np.nan
        avg_win = wins["PnL_Currency"].mean() if not wins.empty else 0
        avg_loss = losses["PnL_Currency"].mean() if not losses.empty else 0
        achieved_rr = np.nan
        if avg_loss < 0:
            achieved_rr = avg_win/abs(avg_loss)
            if total_pnl < 0 and not np.isnan(achieved_rr):
                achieved_rr = -achieved_rr
        return {
            "Total Trades": total_trades,
            "Total PnL": round(total_pnl, global_decimals),
            "Win Rate (%)": round(win_rate, global_decimals),
            "Avg PnL": round(avg_pnl, global_decimals) if not np.isnan(avg_pnl) else np.nan,
            "Median PnL": round(median_pnl, global_decimals) if not np.isnan(median_pnl) else np.nan,
            "Std Dev PnL": round(std_pnl, global_decimals) if not np.isnan(std_pnl) else np.nan,
            "Gross Profit": round(gross_profit, global_decimals),
            "Gross Loss": round(gross_loss, global_decimals),
            "Profit Factor": round(profit_factor, global_decimals) if not np.isnan(profit_factor) else np.nan,
            "Avg Holding Time (hrs)": round(avg_holding, global_decimals) if not np.isnan(avg_holding) else np.nan,
            "Avg PnL/Trade/Lot": round(avg_pnl_per_lot, global_decimals) if not np.isnan(avg_pnl_per_lot) else np.nan,
            "Achieved RR": round(achieved_rr, global_decimals) if not np.isnan(achieved_rr) else np.nan
        }
    summary_dict = {}
    for grp, subset in groups.items():
        summary_dict[grp] = metrics_for(subset)
    summary_df = pd.DataFrame(summary_dict).T.reset_index().rename(columns={"index":"Group"})
    return summary_df

@st.cache_data(show_spinner=False)
def trade_based_sharpe(trades, annual_factor=252):
    if trades.empty or "PnL_Currency" not in trades.columns:
        return np.nan
    if "Entry_Date" in trades.columns and "Exit_Date" in trades.columns:
        durations = (trades["Exit_Date"] - trades["Entry_Date"]).dt.total_seconds()/86400.0
    else:
        durations = np.ones(len(trades))
    daily_returns = []
    for i, row in trades.iterrows():
        pnl = row["PnL_Currency"]
        d = durations.loc[i]
        if d <= 0:
            d = 1/1440
        daily_returns.append(pnl/d)
    s = pd.Series(daily_returns)
    if s.std() == 0:
        return np.nan
    return (s.mean()/s.std()) * math.sqrt(annual_factor)

@st.cache_data(show_spinner=False)
def sortino_ratio(equity_series, annual_factor=252):
    rets = equity_series.pct_change().dropna()
    if len(rets) < 2:
        return np.nan
    downside = rets[rets < 0]
    if downside.std() == 0:
        return np.nan
    return (rets.mean()/downside.std()) * math.sqrt(annual_factor)

@st.cache_data(show_spinner=False)
def calmar_ratio(equity_series):
    if equity_series.empty:
        return np.nan
    start = equity_series.iloc[0]
    end = equity_series.iloc[-1]
    if start <= 0:
        return np.nan
    days = (equity_series.index[-1] - equity_series.index[0]).days
    if days <= 0:
        return np.nan
    ann_return = (end/start)**(365/days)-1
    dd = (equity_series - equity_series.cummax())/equity_series.cummax()
    max_dd = dd.min()
    if max_dd == 0:
        return np.nan
    return ann_return/abs(max_dd)

@st.cache_data(show_spinner=False)
def ulcer_index(equity_series):
    if equity_series.empty:
        return np.nan
    running_max = equity_series.cummax()
    dd = (equity_series - running_max)/running_max
    return math.sqrt((dd**2).mean())

@st.cache_data(show_spinner=False)
def omega_ratio(equity_series, threshold=0):
    rets = equity_series.pct_change().dropna()
    if len(rets) < 1:
        return np.nan
    gains = rets[rets > threshold].sum()
    losses = rets[rets < threshold].sum()
    if losses >= 0:
        return np.nan
    return gains/abs(losses)

@st.cache_data(show_spinner=False)
def kelly_fraction(trade_df):
    if trade_df.empty or "PnL_Currency" not in trade_df.columns:
        return np.nan
    total = len(trade_df)
    wins = trade_df[trade_df["PnL_Currency"] > 0]
    losses = trade_df[trade_df["PnL_Currency"] < 0]
    p = len(wins)/total if total>0 else 0
    avg_win = wins["PnL_Currency"].mean() if not wins.empty else 0
    avg_loss = losses["PnL_Currency"].mean() if not losses.empty else 0
    if avg_win==0 or avg_loss==0:
        return np.nan
    r = abs(avg_loss)/avg_win
    q = 1-p
    numerator = p - q/r
    return numerator/r if r!=0 else np.nan

#####################################
# SIDEBAR: Data Input & Global Settings
#####################################

st.sidebar.header("Trade Filtering Options")
exclude_zero_entry_size = st.sidebar.checkbox("Exclude trades with zero entry size", value=True)

st.sidebar.header("Data Input Options")
input_mode = st.sidebar.radio("Select Input Mode:", ["Multi-File Upload", "Folder Path"])

strategy_data = {}
if input_mode == "Multi-File Upload":
    uploaded_files = st.sidebar.file_uploader("Upload Trade List CSV(s)", type=["csv"], accept_multiple_files=True, key="multi_upload")
    if uploaded_files:
        for file in uploaded_files:
            try:
                df = load_csv_from_file(file)
                if "Entry_Date" in df.columns:
                    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
                if "Exit_Date" in df.columns:
                    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
                if "Entry_Date" in df.columns and "Exit_Date" in df.columns:
                    df["Duration_Hours"] = (df["Exit_Date"] - df["Entry_Date"]).dt.total_seconds()/3600
                if ("Entry_Size" in df.columns) and ("PnL_Currency" in df.columns) and ("PnL_per_Lot" not in df.columns):
                    df["PnL_per_Lot"] = df["PnL_Currency"]/df["Entry_Size"]
                if exclude_zero_entry_size and "Entry_Size" in df.columns:
                    df = df[df["Entry_Size"] != 0]
                strategy_data[file.name] = df
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
elif input_mode == "Folder Path":
    folder_path = st.sidebar.text_input("Enter folder path", value="")
    if folder_path:
        files = glob.glob(os.path.join(folder_path, "*.csv"))
        if files:
            for path in files:
                try:
                    df = pd.read_csv(path)
                    if "Entry_Date" in df.columns:
                        df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
                    if "Exit_Date" in df.columns:
                        df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
                    if "Entry_Date" in df.columns and "Exit_Date" in df.columns:
                        df["Duration_Hours"] = (df["Exit_Date"] - df["Entry_Date"]).dt.total_seconds()/3600
                    if ("Entry_Size" in df.columns) and ("PnL_Currency" in df.columns) and ("PnL_per_Lot" not in df.columns):
                        df["PnL_per_Lot"] = df["PnL_Currency"]/df["Entry_Size"]
                    if exclude_zero_entry_size and "Entry_Size" in df.columns:
                        df = df[df["Entry_Size"] != 0]
                    strategy_data[os.path.basename(path)] = df
                except Exception as e:
                    st.error(f"Error processing {path}: {e}")
        else:
            st.warning("No CSV files found in the folder.")

global_decimals = st.sidebar.number_input("Decimal Precision", min_value=0, max_value=10, value=3, step=1)

#####################################
# SIDEBAR: Combine Multiple Sheets (New Feature)
#####################################
st.sidebar.header("Combine Multiple Sheets")
combine_sheets = st.sidebar.checkbox("Combine trade sheets into one?", value=False)
if combine_sheets and len(strategy_data) > 1:
    sort_option = st.sidebar.radio("Sort combined trades by:", ["Entry_Date", "Exit_Date"])
    # Combine all data
    all_dfs = list(strategy_data.values())
    combined_df = pd.concat(all_dfs, ignore_index=True)
    if sort_option in combined_df.columns:
        combined_df.sort_values(sort_option, inplace=True)
    # Name the combined strategy
    strategy_data["Combined"] = combined_df

# Now that combine is done, build the list of strategies
all_strategies = list(strategy_data.keys())

# Sidebar: Strategy Selection with search filtering
search_term = st.sidebar.text_input("Search Strategy Name", value="")
filtered_strategies = [name for name in all_strategies if search_term.lower() in name.lower()]
if not filtered_strategies:
    filtered_strategies = all_strategies

if strategy_data:
    selected_strategy = st.sidebar.selectbox("Select Strategy for Individual Analysis", options=filtered_strategies)
    active_df = strategy_data[selected_strategy]
else:
    active_df = pd.DataFrame()

#####################################
# SIDEBAR: Clustering Options
#####################################
if not active_df.empty:
    st.sidebar.header("Clustering Options")
    all_cols = list(active_df.columns)
    default_features = []
    for col in ["PnL_Currency", "Duration_Hours", "PnL_per_Lot"]:
        if col in all_cols:
            default_features.append(col)
    clustering_features = st.sidebar.multiselect("Select Clustering Features",
                                                   options=all_cols,
                                                   default=default_features)
    n_clusters = st.sidebar.number_input("Number of Clusters", min_value=2, max_value=10, value=3, step=1)
    possible_classifications = ["Entry_Type", "Volatility_Zone", "Exit_Method", "Direction", "Trend_ID"]
    available_classifications = [col for col in possible_classifications if col in active_df.columns]
    classification_dimensions = st.sidebar.multiselect("Select Classification Dimensions",
                                                       options=available_classifications,
                                                       default=[])
else:
    clustering_features = []
    classification_dimensions = []
    n_clusters = 3

def perform_clustering_custom(df, features, n_clusters=3):
    df_encoded = encode_features(df, features)
    available_features = [f for f in features if f in df_encoded.columns]
    if not available_features:
        return None, None
    X = df_encoded[available_features].dropna()
    if X.empty:
        return None, None
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    return X, clusters

#####################################
# MAIN TABS
#####################################
individual_tab_names = [
    "Trade List",
    "Performance Summary",
    "Advanced Metrics",
    "Entry Size Stats",
    "Aggregated Stats",
    "Extra Info",
    "Trend Group Stats",
    "Drawdown & Recovery",
    "Profit Factor Breakdown",
    "Time-of-Day Analysis",
    "Clustering Analysis"
]
comparison_tab_names = [
    "Strategy Comparison Summary",
    "Equity Curve Comparison",
    "Advanced Metrics Comparison",
    "Strategy Ranking & Composite Score",
    "Entry Size Stats Comparison Across Strategies",
    "Strategy Judgement Ratio Comparison"
]
all_tab_names = individual_tab_names + comparison_tab_names
tabs = st.tabs(all_tab_names)

#####################################
# 1) TRADE LIST TAB (Individual)
#####################################
with tabs[0]:
    st.header("Trade List")
    if not active_df.empty:
        st.markdown("##### Filter Your Trades")
        if "Entry_Date" in active_df.columns:
            min_entry = active_df["Entry_Date"].min()
            max_entry = active_df["Entry_Date"].max()
            sel_entry_date = st.date_input("Select Entry Date Range", [min_entry.date(), max_entry.date()])
        if "Exit_Date" in active_df.columns:
            min_exit = active_df["Exit_Date"].min()
            max_exit = active_df["Exit_Date"].max()
            sel_exit_date = st.date_input("Select Exit Date Range", [min_exit.date(), max_exit.date()])
        vol_options = active_df["Volatility_Zone"].dropna().unique().tolist() if "Volatility_Zone" in active_df.columns else []
        sel_vol = st.multiselect("Volatility Zones", options=vol_options, default=vol_options)
        if "Duration_Hours" in active_df.columns:
            dur_min = float(active_df["Duration_Hours"].min())
            dur_max = float(active_df["Duration_Hours"].max())
            sel_dur = st.slider("Trade Duration (Hours)", min_value=dur_min, max_value=dur_max, value=(dur_min, dur_max))
        else:
            sel_dur = (None, None)
        if "Entry_Price" in active_df.columns:
            price_min = float(active_df["Entry_Price"].min())
            price_max = float(active_df["Entry_Price"].max())
            sel_entry_price = st.slider("Entry Price Range", min_value=price_min, max_value=price_max, value=(price_min, price_max))
        if "Exit_Price" in active_df.columns:
            exit_price_min = float(active_df["Exit_Price"].min())
            exit_price_max = float(active_df["Exit_Price"].max())
            sel_exit_price = st.slider("Exit Price Range", min_value=exit_price_min, max_value=exit_price_max, value=(exit_price_min, exit_price_max))

        pnl_min = float(active_df["PnL_Currency"].min())
        pnl_max = float(active_df["PnL_Currency"].max())
        sel_pnl = st.slider("PnL Range", min_value=pnl_min, max_value=pnl_max, value=(pnl_min, pnl_max))

        filtered = active_df.copy()
        if sel_vol and "Volatility_Zone" in filtered.columns:
            filtered = filtered[filtered["Volatility_Zone"].isin(sel_vol)]
        if sel_dur[0] is not None:
            filtered = filtered[(filtered["Duration_Hours"]>=sel_dur[0]) & (filtered["Duration_Hours"]<=sel_dur[1])]
        filtered = filtered[(filtered["PnL_Currency"]>=sel_pnl[0]) & (filtered["PnL_Currency"]<=sel_pnl[1])]
        if "Entry_Date" in filtered.columns and len(sel_entry_date)==2:
            filtered = filtered[(filtered["Entry_Date"].dt.date >= sel_entry_date[0]) & (filtered["Entry_Date"].dt.date <= sel_entry_date[1])]
        if "Exit_Date" in filtered.columns and len(sel_exit_date)==2:
            filtered = filtered[(filtered["Exit_Date"].dt.date >= sel_exit_date[0]) & (filtered["Exit_Date"].dt.date <= sel_exit_date[1])]
        if "Entry_Price" in filtered.columns:
            filtered = filtered[(filtered["Entry_Price"]>=sel_entry_price[0]) & (filtered["Entry_Price"]<=sel_entry_price[1])]
        if "Exit_Price" in filtered.columns:
            filtered = filtered[(filtered["Exit_Price"]>=sel_exit_price[0]) & (filtered["Exit_Price"]<=sel_exit_price[1])]

        display_aggrid_table(filtered, key="ind_trade_list")
        st.download_button("Download Filtered Trades", filtered.to_csv(index=False).encode("utf-8"),
                           file_name="filtered_trades.csv", mime="text/csv")
        st.markdown("**Intelligent Summary**")
        st.write(f"**{len(active_df)} trades** loaded; average PnL ~ {active_df['PnL_Currency'].mean():.{global_decimals}f}.")
        commentary = interpret_trade_list(active_df)
        st.subheader("Automated Research Commentary")
        st.write(commentary)
        if "Entry_Price" in active_df.columns and "Exit_Price" in active_df.columns:
            fig_scatter_price = px.scatter(active_df, x="Entry_Price", y="Exit_Price",
                                           color="Direction" if "Direction" in active_df.columns else None,
                                           title="Entry Price vs Exit Price", template="plotly_dark")
            st.plotly_chart(fig_scatter_price, use_container_width=True)
        if "PnL_per_Lot" in active_df.columns:
            fig_violin = px.violin(active_df, y="PnL_per_Lot", box=True, points="all", template="plotly_dark",
                                   title="Distribution of PnL per Lot")
            st.plotly_chart(fig_violin, use_container_width=True)
    else:
        st.write("No trade data loaded.")

#####################################
# 2) PERFORMANCE SUMMARY TAB (Individual)
#####################################
with tabs[1]:
    st.header("Performance Summary")
    if not active_df.empty and "Exit_Date" in active_df.columns:
        trades_sorted = active_df.sort_values("Exit_Date")
        total_trades = len(trades_sorted)
        total_pnl = trades_sorted["PnL_Currency"].sum()
        winning = trades_sorted[trades_sorted["PnL_Currency"] > 0]
        losing = trades_sorted[trades_sorted["PnL_Currency"] < 0]
        n_wins = len(winning)
        win_rate = (n_wins/total_trades*100) if total_trades>0 else 0
        avg_pnl_trade = trades_sorted["PnL_Currency"].mean() if total_trades>0 else 0
        gross_profit = winning["PnL_Currency"].sum()
        gross_loss = losing["PnL_Currency"].sum()
        pf = gross_profit/abs(gross_loss) if gross_loss<0 else np.nan
        avg_win = winning["PnL_Currency"].mean() if not winning.empty else 0
        avg_loss = losing["PnL_Currency"].mean() if not losing.empty else 0
        achieved_rr = avg_win/abs(avg_loss) if avg_loss < 0 else np.nan
        trades_sorted["Cumulative_PnL"] = trades_sorted["PnL_Currency"].cumsum()
        peak = trades_sorted["Cumulative_PnL"].cummax()
        drawdowns = trades_sorted["Cumulative_PnL"] - peak
        max_dd = drawdowns.min() if not drawdowns.empty else 0
        if total_trades>0:
            start_date = trades_sorted["Exit_Date"].iloc[0]
            end_date = trades_sorted["Exit_Date"].iloc[-1]
            total_days = (end_date - start_date).days
            data_years = total_days/365.0 if total_days>0 else 1
        else:
            data_years = 1
        strat_judgment_ratio = (total_pnl/abs(max_dd))/data_years if max_dd<0 else np.nan
        Pnl_Max_dd_ratio = (total_pnl/abs(max_dd)) if max_dd<0 else np.nan
        cons_wins = cons_losses = max_cons_wins = max_cons_losses = 0
        for _, row in trades_sorted.iterrows():
            if row["PnL_Currency"] > 0:
                cons_wins += 1
                max_cons_wins = max(max_cons_wins, cons_wins)
                cons_losses = 0
            else:
                cons_losses += 1
                max_cons_losses = max(max_cons_losses, cons_losses)
                cons_wins = 0

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Trades", total_trades)
        col2.metric("Win Rate (%)", f"{win_rate:.{global_decimals}f}")
        col3.metric("Achieved RR", f"{achieved_rr:.{global_decimals}f}" if not np.isnan(achieved_rr) else "N/A")
        col4.metric("Consec. Wins", max_cons_wins)
        col5.metric("Consec. Losses", max_cons_losses)
        col6, col7, col8, col9, col10 = st.columns(5)
        col6.metric("Total PnL", f"{total_pnl:.{global_decimals}f}")
        col7.metric("Avg PnL/Trade", f"{avg_pnl_trade:.{global_decimals}f}")
        col8.metric("Profit Factor", f"{pf:.{global_decimals}f}" if not np.isnan(pf) else "N/A")
        col9.metric("Gross Profit", f"{gross_profit:.{global_decimals}f}")
        col10.metric("Gross Loss", f"{gross_loss:.{global_decimals}f}")
        col11, col12, col13, col14, col15 = st.columns(5)
        col11.metric("Max Drawdown", f"{max_dd:.{global_decimals}f}")
        col12.metric("Strategy Judgment Ratio", f"{strat_judgment_ratio:.{global_decimals}f}" if not np.isnan(strat_judgment_ratio) else "N/A")
        col13.metric("PnL & Max DD Ratio", f"{Pnl_Max_dd_ratio:.{global_decimals}f}" if not np.isnan(Pnl_Max_dd_ratio) else "N/A")
        col14.metric("Peak Equity", f"{peak.max():.{global_decimals}f}")
        if not drawdowns.empty:
            col15.metric("Current Drawdown", f"{drawdowns.iloc[-1]:.{global_decimals}f}")
        else:
            col15.metric("Current Drawdown", "N/A")

        st.subheader("Equity Curve with Moving Average")
        fig_eq = px.line(trades_sorted, x="Exit_Date", y="Cumulative_PnL", template="plotly_dark", title="Equity Curve")
        ma = moving_average(trades_sorted["Cumulative_PnL"])
        fig_eq.add_scatter(x=trades_sorted["Exit_Date"], y=ma, mode='lines', name="20-Period MA")
        st.plotly_chart(fig_eq, use_container_width=True)

        st.subheader("Group Summary")
        group_summary = compute_group_summary(trades_sorted, global_decimals=global_decimals)
        display_aggrid_table(group_summary, key="group_summary")
        if "Group" in group_summary.columns and "Profit Factor" in group_summary.columns:
            fig_pf = px.bar(group_summary, x="Group", y="Profit Factor", template="plotly_dark", title="Profit Factor by Group")
            st.plotly_chart(fig_pf, use_container_width=True)
        if "Group" in group_summary.columns and "Win Rate (%)" in group_summary.columns:
            fig_wr = px.bar(group_summary, x="Group", y="Win Rate (%)", template="plotly_dark", title="Win Rate by Group")
            st.plotly_chart(fig_wr, use_container_width=True)

        fig_density = px.histogram(trades_sorted, x="PnL_Currency", nbins=30, histnorm='density', template="plotly_dark", title="PnL Density")
        st.plotly_chart(fig_density, use_container_width=True)
        st.markdown("**Intelligent Summary**")
        st.write(f"Total {total_trades} trades, Total PnL {total_pnl:.{global_decimals}f}, Win Rate {win_rate:.{global_decimals}f}%.")
        commentary = interpret_performance_summary(win_rate, pf, achieved_rr, max_dd, strat_judgment_ratio,
                                                   total_trades, total_pnl, avg_pnl_trade,
                                                   max_cons_losses, max_cons_wins,
                                                   pnl_series=trades_sorted["PnL_Currency"],
                                                   trade_dates=trades_sorted["Exit_Date"])
        st.subheader("Automated Research Commentary")
        st.write(commentary)
    else:
        st.write("No trades loaded for summary.")

#####################################
# 3) ADVANCED METRICS TAB
#####################################
with tabs[2]:
    st.header("Advanced Metrics")
    if not active_df.empty:
        ts = active_df.sort_values("Exit_Date").copy()
        ts["Cumulative_PnL"] = ts["PnL_Currency"].cumsum()
        eq_series = ts["Cumulative_PnL"]
        eq_series.index = ts["Exit_Date"]
        s_ratio = trade_based_sharpe(ts, annual_factor=252)
        sort_val = sortino_ratio(eq_series, annual_factor=252)
        cal_val = calmar_ratio(eq_series)
        ulcer_val = ulcer_index(eq_series)
        omega_val = omega_ratio(eq_series)
        base_kelly = kelly_fraction(ts)
        cvar_val = compute_cvar(ts, confidence=0.95)
        var_val = compute_var(ts, confidence=0.95)
        cdar_val = compute_cdar(ts)
        skew_val, kurt_val = get_skewness_kurtosis(ts)

        col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
        col1.metric("Sharpe", f"{s_ratio:.{global_decimals}f}" if not np.isnan(s_ratio) else "N/A")
        col2.metric("Sortino", f"{sort_val:.{global_decimals}f}" if not np.isnan(sort_val) else "N/A")
        col3.metric("Calmar", f"{cal_val:.{global_decimals}f}" if not np.isnan(cal_val) else "N/A")
        col4.metric("Ulcer", f"{ulcer_val:.{global_decimals}f}" if not np.isnan(ulcer_val) else "N/A")
        col5.metric("Omega", f"{omega_val:.{global_decimals}f}" if not np.isnan(omega_val) else "N/A")
        col6.metric("Kelly", f"{base_kelly:.{global_decimals}f}" if not np.isnan(base_kelly) else "N/A")
        col7.metric("CVaR", f"{cvar_val:.{global_decimals}f}" if not np.isnan(cvar_val) else "N/A")
        col8.metric("VaR", f"{var_val:.{global_decimals}f}" if not np.isnan(var_val) else "N/A")
        col9.metric("CDaR", f"{cdar_val:.{global_decimals}f}" if not np.isnan(cdar_val) else "N/A")

        st.subheader("PnL Distribution Analysis")
        fig_hist = px.histogram(ts, x="PnL_Currency", nbins=30, template="plotly_dark", title="PnL Histogram")
        st.plotly_chart(fig_hist, use_container_width=True)
        fig_box = px.box(ts, y="PnL_Currency", template="plotly_dark", title="PnL Box Plot")
        st.plotly_chart(fig_box, use_container_width=True)
        fig_kde = px.density_contour(ts, x="PnL_Currency", template="plotly_dark", title="PnL Density Contour")
        st.plotly_chart(fig_kde, use_container_width=True)
        st.markdown("**Intelligent Summary**")
        st.write(f"Risk metrics: Sharpe {s_ratio:.{global_decimals}f}, Sortino {sort_val:.{global_decimals}f}, Calmar {cal_val:.{global_decimals}f}, CVaR {cvar_val:.{global_decimals}f}.")
        commentary = interpret_advanced_metrics(s_ratio, sort_val, cal_val, ulcer_val, omega_val, base_kelly, eq_series)
        st.subheader("Automated Research Commentary")
        st.write(commentary)
    else:
        st.write("No trade data available.")

#####################################
# 4) ENTRY SIZE STATS TAB (Individual)
#####################################
with tabs[3]:
    st.header("Entry Size Stats")
    if not active_df.empty and "Entry_Size" in active_df.columns:
        group_es = active_df.groupby("Entry_Size").agg(
            Total_Trades=("PnL_Currency", "count"),
            Total_PnL=("PnL_Currency", "sum"),
            Avg_PnL=("PnL_Currency", "mean"),
            Median_PnL=("PnL_Currency", "median"),
            Std_PnL=("PnL_Currency", "std"),
            Gross_Profit=("PnL_Currency", lambda x: x[x > 0].sum()),
            Gross_Loss=("PnL_Currency", lambda x: x[x < 0].sum())
        ).reset_index()
        group_es["Profit_Factor"] = group_es.apply(
            lambda r: r["Gross_Profit"] / abs(r["Gross_Loss"]) if r["Gross_Loss"] < 0 else np.nan, axis=1
        )
        if "PnL_per_Lot" in active_df.columns:
            group_es["Avg PnL per Trade"] = active_df.groupby("Entry_Size")["PnL_Currency"].mean().values
            group_es["Avg PnL per Lot"] = active_df.groupby("Entry_Size")["PnL_per_Lot"].mean().values
        display_aggrid_table(group_es, key="entry_size_table")
        fig_bar = px.bar(group_es, x="Entry_Size", y="Total_PnL", template="plotly_dark", title="Total PnL by Entry Size")
        st.plotly_chart(fig_bar, use_container_width=True)
        fig_bar2 = px.bar(group_es, x="Entry_Size", y="Avg PnL per Trade", template="plotly_dark", title="Avg PnL per Trade by Entry Size")
        st.plotly_chart(fig_bar2, use_container_width=True)
        if "PnL_per_Lot" in active_df.columns:
            fig_bar3 = px.bar(group_es, x="Entry_Size", y="Avg PnL per Lot", template="plotly_dark", title="Avg PnL per Lot by Entry Size")
            st.plotly_chart(fig_bar3, use_container_width=True)

        st.subheader("Detailed Breakdown (Entry_Size x Volatility_Zone x Exit_Method)")
        if "Volatility_Zone" in active_df.columns and "Exit_Method" in active_df.columns:
            if "PnL_per_Lot" in active_df.columns:
                det_es = active_df.groupby(["Entry_Size", "Volatility_Zone", "Exit_Method"]).agg(
                    Total_Trades=("PnL_Currency", "count"),
                    Total_PnL=("PnL_Currency", "sum"),
                    Avg_PnL=("PnL_Currency", "mean"),
                    Avg_PnL_per_Lot=("PnL_per_Lot", "mean")
                ).reset_index()
            else:
                det_es = active_df.groupby(["Entry_Size", "Volatility_Zone", "Exit_Method"]).agg(
                    Total_Trades=("PnL_Currency", "count"),
                    Total_PnL=("PnL_Currency", "sum"),
                    Avg_PnL=("PnL_Currency", "mean")
                ).reset_index()
            display_aggrid_table(det_es, key="entry_size_detail")
            fig_sun = px.sunburst(det_es, path=["Entry_Size", "Volatility_Zone", "Exit_Method"], values="Total_Trades",
                                  color="Avg_PnL", title="Sunburst: Entry Size x VolZone x ExitMethod",
                                  template="plotly_dark")
            st.plotly_chart(fig_sun, use_container_width=True, key="fig_sun_es")

        st.subheader("Long vs. Short Classification")
        if "Direction" in active_df.columns:
            if "PnL_per_Lot" in active_df.columns:
                long_es = active_df[active_df["Direction"] == 1].groupby("Entry_Size").agg(
                    Trade_Count=("PnL_Currency", "count"),
                    Total_PnL=("PnL_Currency", "sum"),
                    Avg_PnL=("PnL_Currency", "mean"),
                    Avg_PnL_per_Lot=("PnL_per_Lot", "mean")
                ).reset_index()
                short_es = active_df[active_df["Direction"] == -1].groupby("Entry_Size").agg(
                    Trade_Count=("PnL_Currency", "count"),
                    Total_PnL=("PnL_Currency", "sum"),
                    Avg_PnL=("PnL_Currency", "mean"),
                    Avg_PnL_per_Lot=("PnL_per_Lot", "mean")
                ).reset_index()
            else:
                long_es = active_df[active_df["Direction"] == 1].groupby("Entry_Size").agg(
                    Trade_Count=("PnL_Currency", "count"),
                    Total_PnL=("PnL_Currency", "sum"),
                    Avg_PnL=("PnL_Currency", "mean")
                ).reset_index()
                short_es = active_df[active_df["Direction"] == -1].groupby("Entry_Size").agg(
                    Trade_Count=("PnL_Currency", "count"),
                    Total_PnL=("PnL_Currency", "sum"),
                    Avg_PnL=("PnL_Currency", "mean")
                ).reset_index()
            st.markdown("**Long Trades**")
            display_aggrid_table(long_es, key="long_trades_es")
            st.markdown("**Short Trades**")
            display_aggrid_table(short_es, key="short_trades_es")
    else:
        st.write("No 'Entry_Size' data available.")

#####################################
# 5) AGGREGATED STATS TAB (Individual)
#####################################
with tabs[4]:
    st.header("Aggregated Stats")
    if not active_df.empty and "Exit_Date" in active_df.columns:
        freq = st.selectbox("Grouping Frequency", ["Day", "Week", "Month", "Quarter", "Year"])
        df_agg = active_df.copy()
        df_agg["Exit_Date"] = pd.to_datetime(df_agg["Exit_Date"])
        if freq == "Day":
            df_agg["Period"] = df_agg["Exit_Date"].dt.strftime("%Y-%m-%d")
        elif freq == "Week":
            df_agg["Period"] = df_agg["Exit_Date"].dt.to_period("W").astype(str)
        elif freq == "Month":
            df_agg["Period"] = df_agg["Exit_Date"].dt.to_period("M").astype(str)
        elif freq == "Quarter":
            df_agg["Period"] = df_agg["Exit_Date"].dt.to_period("Q").astype(str)
        else:
            df_agg["Period"] = df_agg["Exit_Date"].dt.year.astype(str)
        agg_res = df_agg.groupby("Period").agg(
            Total_Profit=("PnL_Currency", "sum"),
            Avg_PnL=("PnL_Currency", "mean"),
            Total_Trades=("PnL_Currency", "count")
        ).reset_index()
        display_aggrid_table(agg_res, key="aggregated_stats")
        fig_agg = px.bar(agg_res, x="Period", y="Total_Profit", template="plotly_dark", title=f"Total Profit by {freq}")
        st.plotly_chart(fig_agg, use_container_width=True)
        agg_res["Upper"] = agg_res["Total_Profit"] * 1.1
        agg_res["Lower"] = agg_res["Total_Profit"] * 0.9
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=agg_res["Period"], y=agg_res["Total_Profit"], mode='lines+markers', name='Total Profit'))
        fig_line.add_trace(go.Scatter(x=agg_res["Period"], y=agg_res["Upper"], fill=None, mode='lines', line_color='lightgrey', name='Upper Band'))
        fig_line.add_trace(go.Scatter(x=agg_res["Period"], y=agg_res["Lower"], fill='tonexty', mode='lines', line_color='lightgrey', name='Lower Band'))
        fig_line.update_layout(title=f"Aggregated Profit with Confidence Bands ({freq})", template="plotly_dark")
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown("**Intelligent Summary**")
        st.write(f"Aggregating by {freq} reveals seasonal and cyclical performance trends.")
        commentary = interpret_aggregated_stats(agg_res)
        st.subheader("Automated Research Commentary")
        st.write(commentary)
    else:
        st.write("No data available for aggregation.")

#####################################
# 6) EXTRA INFO TAB (Individual)
#####################################
with tabs[5]:
    st.header("Extra Info")
    if not active_df.empty:
        st.subheader("PnL & Duration Distributions")
        fig_hist = px.histogram(active_df, x="PnL_Currency", nbins=30, template="plotly_dark", title="Trade PnL Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)
        fig_box = px.box(active_df, y="PnL_Currency", template="plotly_dark", title="Trade PnL Box Plot")
        st.plotly_chart(fig_box, use_container_width=True)
        if "Duration_Hours" in active_df.columns:
            fig_dur = px.histogram(active_df, x="Duration_Hours", nbins=30, template="plotly_dark", title="Trade Duration Distribution")
            st.plotly_chart(fig_dur, use_container_width=True)
            fig_scatter = px.scatter(active_df, x="Duration_Hours", y="PnL_Currency", template="plotly_dark", title="Duration vs. PnL")
            st.plotly_chart(fig_scatter, use_container_width=True)
        num_cols = active_df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            corr_matrix = correlation_matrix(active_df, num_cols)
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", template="plotly_dark", title="Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)
        commentary = interpret_extra_info(durations=active_df["Duration_Hours"] if "Duration_Hours" in active_df.columns else None,
                                          trade_pnl=active_df["PnL_Currency"])
        st.subheader("Automated Research Commentary")
        st.write(commentary)
    else:
        st.write("No extra information available.")

#####################################
# 7) TREND GROUP STATS TAB (Individual)
#####################################
with tabs[6]:
    st.header("Trend Group Stats")
    if not active_df.empty and "Trend_ID" in active_df.columns:
        df_trend = active_df.copy()
        df_trend["Trend_ID"] = df_trend["Trend_ID"].astype(str)
        trend_stats = df_trend.groupby("Trend_ID").agg({
            "PnL_Currency": ["sum", "mean", "count"],
            "Entry_Date": "min",
            "Exit_Date": "max"
        })
        trend_stats.columns = ["Total Trend PnL", "Avg Trade PnL", "Num Trades", "Trend Start", "Trend End"]
        trend_stats = trend_stats.reset_index()
        trend_stats["Trend Start"] = pd.to_datetime(trend_stats["Trend Start"])
        trend_stats["Trend End"] = pd.to_datetime(trend_stats["Trend End"])
        trend_stats["Trend Duration (Days)"] = (trend_stats["Trend End"] - trend_stats["Trend Start"]).dt.days
        trend_stats["Trend Win"] = (trend_stats["Total Trend PnL"] > 0).astype(int)
        display_aggrid_table(trend_stats, key="trend_stats")
        fig_bar = px.bar(trend_stats, x="Trend_ID", y="Total Trend PnL", template="plotly_dark", title="Total Trend PnL")
        st.plotly_chart(fig_bar, use_container_width=True)
        fig_scatter = px.scatter(trend_stats, x="Trend Duration (Days)", y="Total Trend PnL",
                                 size="Num Trades", template="plotly_dark", title="Duration vs. Trend PnL")
        st.plotly_chart(fig_scatter, use_container_width=True)
        commentary = interpret_trend_group_stats(trend_stats)
        st.subheader("Automated Research Commentary")
        st.write(commentary)
    else:
        st.write("No trend data available.")

#####################################
# 8) DRAWDOWN & RECOVERY TAB (Individual)
#####################################
with tabs[7]:
    st.header("Drawdown & Recovery")
    if not active_df.empty:
        dd_df = active_df.sort_values("Exit_Date").copy()
        dd_df["Cumulative_PnL"] = dd_df["PnL_Currency"].cumsum()
        dd_df["Running_Max"] = dd_df["Cumulative_PnL"].cummax()
        dd_df["Drawdown"] = dd_df["Cumulative_PnL"] - dd_df["Running_Max"]
        dd_df["Drawdown_%"] = np.where(dd_df["Running_Max"]!=0, dd_df["Drawdown"]/dd_df["Running_Max"], 0)
        fig_dd = px.line(dd_df, x="Exit_Date", y="Drawdown", template="plotly_dark", title="Drawdown Curve")
        st.plotly_chart(fig_dd, use_container_width=True)
        st.subheader("Top 5 Drawdowns")
        dd_sorted = dd_df.sort_values("Drawdown").head(5)
        display_aggrid_table(dd_sorted[["Exit_Date", "Cumulative_PnL", "Running_Max", "Drawdown", "Drawdown_%"]], key="top5_dd")
        if "Volatility_Zone" in dd_df.columns:
            dd_df["Date_Str"] = dd_df["Exit_Date"].dt.strftime("%Y-%m-%d")
            pivot_dd = dd_df.pivot_table(index="Date_Str", columns="Volatility_Zone", values="Drawdown_%", aggfunc="mean")
            fig_heat = px.imshow(pivot_dd.T, aspect="auto", color_continuous_scale="RdBu_r",
                                 title="Drawdown (%) Heatmap", labels={'color':'Drawdown %'}, origin="lower")
            st.plotly_chart(fig_heat, use_container_width=True)
        commentary = interpret_drawdown_analysis(dd_df)
        st.subheader("Automated Research Commentary")
        st.write(commentary)
    else:
        st.write("No drawdown data available.")

#####################################
# 9) PROFIT FACTOR BREAKDOWN TAB (Individual)
#####################################
with tabs[8]:
    st.header("Profit Factor Breakdown")
    if not active_df.empty:
        wins = active_df[active_df["PnL_Currency"] > 0]
        losses = active_df[active_df["PnL_Currency"] < 0]
        gp = wins["PnL_Currency"].sum()
        gl = losses["PnL_Currency"].sum()
        pf_all = gp/abs(gl) if gl < 0 else np.nan
        st.metric("Overall Profit Factor", f"{pf_all:.{global_decimals}f}" if not np.isnan(pf_all) else "N/A")
        pf_dir = pf_vol = pf_exit = None
        if "Direction" in active_df.columns:
            pf_dir = active_df.groupby("Direction")["PnL_Currency"].agg(
                Gross_Profit=lambda x: x[x>0].sum(),
                Gross_Loss=lambda x: x[x<0].sum()
            ).reset_index()
            pf_dir["Profit_Factor"] = pf_dir.apply(lambda r: r["Gross_Profit"]/abs(r["Gross_Loss"]) if r["Gross_Loss"]<0 else np.nan, axis=1)
            st.subheader("By Direction")
            display_aggrid_table(pf_dir, key="pf_dir")
        if "Volatility_Zone" in active_df.columns:
            pf_vol = active_df.groupby("Volatility_Zone")["PnL_Currency"].agg(
                Gross_Profit=lambda x: x[x>0].sum(),
                Gross_Loss=lambda x: x[x<0].sum()
            ).reset_index()
            pf_vol["Profit_Factor"] = pf_vol.apply(lambda r: r["Gross_Profit"]/abs(r["Gross_Loss"]) if r["Gross_Loss"]<0 else np.nan, axis=1)
            st.subheader("By Volatility Zone")
            display_aggrid_table(pf_vol, key="pf_vol")
        if "Exit_Method" in active_df.columns:
            pf_exit = active_df.groupby("Exit_Method")["PnL_Currency"].agg(
                Gross_Profit=lambda x: x[x>0].sum(),
                Gross_Loss=lambda x: x[x<0].sum()
            ).reset_index()
            pf_exit["Profit_Factor"] = pf_exit.apply(lambda r: r["Gross_Profit"]/abs(r["Gross_Loss"]) if r["Gross_Loss"]<0 else np.nan, axis=1)
            st.subheader("By Exit Method")
            display_aggrid_table(pf_exit, key="pf_exit")
        st.markdown("**Intelligent Summary**")
        st.write(f"Overall PF ~ {pf_all:.{global_decimals}f}" if not np.isnan(pf_all) else "N/A")
        commentary = interpret_profit_factor_breakdown(pf_dir, pf_vol, pf_exit)
        st.subheader("Automated Research Commentary")
        st.write(commentary)
    else:
        st.write("No trade data available.")

#####################################
# 10) TIME-OF-DAY ANALYSIS TAB (Individual)
#####################################
with tabs[9]:
    st.header("Time-of-Day / Day-of-Week Analysis")
    if not active_df.empty and "Entry_Date" in active_df.columns:
        tdf = active_df.copy()
        tdf["Entry_Hour"] = tdf["Entry_Date"].dt.hour
        tdf["Entry_DayOfWeek"] = tdf["Entry_Date"].dt.day_name()
        st.subheader("PnL by Hour")
        hour_agg = tdf.groupby("Entry_Hour")["PnL_Currency"].agg(["count","sum","mean"]).reset_index()
        display_aggrid_table(hour_agg, key="hour_agg")
        fig_hour = px.bar(hour_agg, x="Entry_Hour", y="sum", template="plotly_dark", title="Total PnL by Hour")
        st.plotly_chart(fig_hour, use_container_width=True)
        st.subheader("PnL by Day of Week")
        dow_agg = tdf.groupby("Entry_DayOfWeek")["PnL_Currency"].agg(["count","sum","mean"]).reset_index()
        display_aggrid_table(dow_agg, key="dow_agg")
        fig_dow = px.bar(dow_agg, x="Entry_DayOfWeek", y="sum", template="plotly_dark", title="Total PnL by Day of Week")
        st.plotly_chart(fig_dow, use_container_width=True)
        pivot_time = tdf.pivot_table(index="Entry_DayOfWeek", columns="Entry_Hour", values="PnL_Currency", aggfunc="sum")
        fig_heat_time = px.imshow(pivot_time, template="plotly_dark", title="Heatmap: PnL by Day & Hour")
        st.plotly_chart(fig_heat_time, use_container_width=True)
        commentary = interpret_time_of_day_analysis(hour_agg, dow_agg)
        st.subheader("Automated Research Commentary")
        st.write(commentary)
    else:
        st.write("No time-of-day data available.")

#####################################
# 11) CLUSTERING ANALYSIS TAB (Individual)
#####################################
with tabs[10]:
    st.header("Trade Clustering & Segmentation")
    if not active_df.empty and clustering_features:
        X, clusters = perform_clustering_custom(active_df, clustering_features, n_clusters=n_clusters)
        if X is not None and clusters is not None:
            df_clustered = active_df.loc[X.index].copy()
            df_clustered["Cluster"] = clusters.astype(str)
            def compute_cluster_metrics(df):
                summary = {
                    "Trade_Count": len(df),
                    "Total_PnL": df["PnL_Currency"].sum(),
                    "Avg_PnL": df["PnL_Currency"].mean(),
                    "Median_PnL": df["PnL_Currency"].median(),
                    "Std_PnL": df["PnL_Currency"].std()
                }
                wins = df[df["PnL_Currency"] > 0]
                losses = df[df["PnL_Currency"] < 0]
                summary["Win_Rate (%)"] = (len(wins) / len(df)*100) if len(df) > 0 else np.nan
                if losses["PnL_Currency"].sum() < 0:
                    summary["Profit_Factor"] = (wins["PnL_Currency"].sum() / abs(losses["PnL_Currency"].sum()))
                else:
                    summary["Profit_Factor"] = np.nan
                if "Duration_Hours" in df.columns:
                    summary["Avg_Duration"] = df["Duration_Hours"].mean()
                return summary
            overall_cluster_summary = df_clustered.groupby("Cluster").apply(compute_cluster_metrics).apply(pd.Series).reset_index()
            st.subheader("Overall Cluster Summary")
            st.dataframe(overall_cluster_summary)
            fig_cluster_count = px.bar(overall_cluster_summary, x="Cluster", y="Trade_Count",
                                       title="Trade Count by Cluster", template="plotly_dark")
            st.plotly_chart(fig_cluster_count, use_container_width=True)
            fig_box_cluster = px.box(df_clustered, x="Cluster", y="PnL_Currency", template="plotly_dark",
                                     title="PnL Distribution by Cluster")
            st.plotly_chart(fig_box_cluster, use_container_width=True)
            available_clusters = sorted(df_clustered["Cluster"].unique())
            selected_clusters = st.multiselect("Select Cluster(s) for Detailed Analysis", 
                                               options=available_clusters,
                                               default=available_clusters)
            if selected_clusters:
                cluster_df = df_clustered[df_clustered["Cluster"].isin(selected_clusters)]
                st.subheader(f"Complete Trade List for Cluster(s) {', '.join(selected_clusters)}")
                display_aggrid_table(cluster_df, key="detailed_cluster_table")
                detailed_metrics = cluster_df.groupby("Cluster").agg(
                    Trade_Count=("PnL_Currency", "count"),
                    Total_PnL=("PnL_Currency", "sum"),
                    Avg_PnL=("PnL_Currency", "mean"),
                    Median_PnL=("PnL_Currency", "median"),
                    Std_PnL=("PnL_Currency", "std")
                ).reset_index()
                if "Duration_Hours" in cluster_df.columns:
                    detailed_metrics["Avg_Duration"] = cluster_df.groupby("Cluster")["Duration_Hours"].mean().values
                st.subheader("Detailed Cluster Metrics")
                st.dataframe(detailed_metrics)
                fig_hist_cluster = px.histogram(cluster_df, x="PnL_Currency", nbins=30, template="plotly_dark",
                                                title="PnL Distribution in Selected Cluster(s)")
                st.plotly_chart(fig_hist_cluster, use_container_width=True)
                if "Duration_Hours" in cluster_df.columns:
                    fig_dur_cluster = px.histogram(cluster_df, x="Duration_Hours", nbins=30, template="plotly_dark",
                                                   title="Duration Distribution in Selected Cluster(s)")
                    st.plotly_chart(fig_dur_cluster, use_container_width=True)
                    fig_scatter_cluster = px.scatter(cluster_df, x="Duration_Hours", y="PnL_Currency", template="plotly_dark",
                                                     title="Duration vs. PnL in Selected Cluster(s)")
                    st.plotly_chart(fig_scatter_cluster, use_container_width=True)
                if "Exit_Date" in cluster_df.columns:
                    cluster_df_sorted = cluster_df.sort_values("Exit_Date").copy()
                    cluster_df_sorted["Cumulative_PnL"] = cluster_df_sorted["PnL_Currency"].cumsum()
                    fig_cluster_eq = px.line(cluster_df_sorted, x="Exit_Date", y="Cumulative_PnL", template="plotly_dark",
                                             title="Cumulative PnL for Selected Cluster(s)")
                    st.plotly_chart(fig_cluster_eq, use_container_width=True)
                if classification_dimensions:
                    st.subheader("Breakdown by Classification Dimensions")
                    for dim in classification_dimensions:
                        if dim in cluster_df.columns:
                            st.markdown(f"**Breakdown by {dim}:**")
                            class_summary = cluster_df.groupby(dim).agg(
                                Trade_Count=("PnL_Currency", "count"),
                                Total_PnL=("PnL_Currency", "sum"),
                                Avg_PnL=("PnL_Currency", "mean"),
                                Median_PnL=("PnL_Currency", "median")
                            ).reset_index()
                            st.dataframe(class_summary)
                            fig_class = px.bar(class_summary, x=dim, y="Total_PnL", template="plotly_dark",
                                               title=f"Total PnL by {dim} in Selected Cluster(s)")
                            st.plotly_chart(fig_class, use_container_width=True)
            else:
                st.info("Please select at least one cluster for detailed analysis.")
            inertia = []
            sil_scores = []
            K_range = range(2, min(10, len(active_df)//5+2))
            for k in K_range:
                kmeans_temp = KMeans(n_clusters=k, random_state=42)
                labels_temp = kmeans_temp.fit_predict(X)
                inertia.append(kmeans_temp.inertia_)
                sil_scores.append(silhouette_score(X, labels_temp))
            fig_elbow = px.line(x=list(K_range), y=inertia, markers=True, template="plotly_dark", title="Elbow Method")
            st.plotly_chart(fig_elbow, use_container_width=True)
            fig_sil = px.line(x=list(K_range), y=sil_scores, markers=True, template="plotly_dark", title="Silhouette Score")
            st.plotly_chart(fig_sil, use_container_width=True)
            pca_components = perform_pca(X)
            pca_df = pd.DataFrame(pca_components, columns=["PC1", "PC2"], index=X.index)
            pca_df["Cluster"] = clusters.astype(str)
            fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
                                 title="PCA Projection of Clusters", template="plotly_dark")
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            st.write("Not enough data to perform clustering.")
    else:
        st.write("No trade data or clustering features available.")

#####################################
# COMPARISON TABS (Only if ≥2 strategies uploaded)
#####################################
if len(strategy_data) > 1:
    base_index = len(individual_tab_names)

    #####################################
    # 12) STRATEGY COMPARISON SUMMARY TAB
    #####################################
    with tabs[base_index]:
        st.header("Strategy Comparison Summary")

        # We gather the same performance summary metrics from each strategy
        # that we show in the Performance Summary tab.
        summary_list = []
        for label, df in strategy_data.items():
            if df.empty or "Exit_Date" not in df.columns:
                continue
            trades_sorted = df.sort_values("Exit_Date")
            total_trades = len(trades_sorted)
            total_pnl = trades_sorted["PnL_Currency"].sum()
            winning = trades_sorted[trades_sorted["PnL_Currency"] > 0]
            losing = trades_sorted[trades_sorted["PnL_Currency"] < 0]
            n_wins = len(winning)
            win_rate = (n_wins/total_trades*100) if total_trades>0 else 0
            avg_pnl_trade = trades_sorted["PnL_Currency"].mean() if total_trades>0 else 0
            gross_profit = winning["PnL_Currency"].sum()
            gross_loss = losing["PnL_Currency"].sum()
            pf = gross_profit/abs(gross_loss) if gross_loss<0 else np.nan
            avg_win = winning["PnL_Currency"].mean() if not winning.empty else 0
            avg_loss = losing["PnL_Currency"].mean() if not losing.empty else 0
            achieved_rr = avg_win/abs(avg_loss) if avg_loss < 0 else np.nan
            trades_sorted["Cumulative_PnL"] = trades_sorted["PnL_Currency"].cumsum()
            peak = trades_sorted["Cumulative_PnL"].cummax()
            drawdowns = trades_sorted["Cumulative_PnL"] - peak
            max_dd = drawdowns.min() if not drawdowns.empty else 0
            if total_trades>0:
                start_date = trades_sorted["Exit_Date"].iloc[0]
                end_date = trades_sorted["Exit_Date"].iloc[-1]
                total_days = (end_date - start_date).days
                data_years = total_days/365.0 if total_days>0 else 1
            else:
                data_years = 1
            strat_judgment_ratio = (total_pnl/abs(max_dd))/data_years if max_dd<0 else np.nan
            Pnl_Max_dd_ratio = (total_pnl/abs(max_dd)) if max_dd<0 else np.nan
            cons_wins = cons_losses = max_cons_wins = max_cons_losses = 0
            for _, row in trades_sorted.iterrows():
                if row["PnL_Currency"] > 0:
                    cons_wins += 1
                    max_cons_wins = max(max_cons_wins, cons_wins)
                    cons_losses = 0
                else:
                    cons_losses += 1
                    max_cons_losses = max(max_cons_losses, cons_losses)
                    cons_wins = 0

            current_dd = drawdowns.iloc[-1] if not drawdowns.empty else np.nan

            summary_list.append({
                "Strategy": label,
                "Total Trades": total_trades,
                "Win Rate (%)": win_rate,
                "Achieved RR": achieved_rr,
                "Consec Wins": max_cons_wins,
                "Consec Losses": max_cons_losses,
                "Total PnL": total_pnl,
                "Avg PnL/Trade": avg_pnl_trade,
                "Profit Factor": pf,
                "Gross Profit": gross_profit,
                "Gross Loss": gross_loss,
                "Max Drawdown": max_dd,
                "Strategy Judgment Ratio": strat_judgment_ratio,
                "PnL & Max DD Ratio": Pnl_Max_dd_ratio,
                "Peak Equity": peak.max() if not peak.empty else np.nan,
                "Current Drawdown": current_dd
            })

           

        comp_df = pd.DataFrame(summary_list)

        st.subheader("Filter Comparison Table by Metrics")
        # Build dynamic filters for each numeric column
        filtered_comp_df = comp_df.copy()
        numeric_cols = [col for col in comp_df.columns if pd.api.types.is_numeric_dtype(comp_df[col]) and col != "Strategy"]
        for col in numeric_cols:
            col_min = float(comp_df[col].min())
            col_max = float(comp_df[col].max())
            val_range = st.slider(f"Filter {col}", min_value=col_min, max_value=col_max, value=(col_min, col_max))
            filtered_comp_df = filtered_comp_df[(filtered_comp_df[col] >= val_range[0]) & (filtered_comp_df[col] <= val_range[1])]

        st.subheader("Comparison Table")
        format_dict = {col: "{:." + str(global_decimals) + "f}" for col in filtered_comp_df.columns if pd.api.types.is_numeric_dtype(filtered_comp_df[col])}
        st.dataframe(filtered_comp_df.style.format(format_dict))

        # Now let's show some bar charts for each metric that might be relevant
        # but only for the filtered subset
        for metric in [
            "Total Trades","Win Rate (%)","Achieved RR","Consec Wins","Consec Losses",
            "Total PnL","Avg PnL/Trade","Profit Factor","Gross Profit","Gross Loss",
            "Max Drawdown","Strategy Judgment Ratio","PnL & Max DD Ratio","Peak Equity","Current Drawdown"
        ]:
            if metric in filtered_comp_df.columns:
                fig = px.bar(filtered_comp_df, x="Strategy", y=metric, template="plotly_dark", title=f"{metric} Comparison")
                st.plotly_chart(fig, use_container_width=True)


    #####################################
    # 13) EQUITY CURVE COMPARISON TAB
    #####################################
    with tabs[base_index+1]:
        st.header("Equity Curve Comparison")
        fig = go.Figure()
        for label, df in strategy_data.items():
            if df.empty or "Exit_Date" not in df.columns:
                continue
            ts = df.sort_values("Exit_Date").copy()
            ts["Cumulative_PnL"] = ts["PnL_Currency"].cumsum()
            fig.add_trace(go.Scatter(x=ts["Exit_Date"], y=ts["Cumulative_PnL"], mode="lines", name=label))
        fig.update_layout(template="plotly_dark", title="Cumulative PnL Across Strategies", xaxis_title="Exit Date", yaxis_title="Cumulative PnL")
        st.plotly_chart(fig, use_container_width=True)

    #####################################
    # 14) ADVANCED METRICS COMPARISON TAB
    #####################################
    with tabs[base_index+2]:
        st.header("Advanced Metrics Comparison")
        comp_metrics = []
        for label, df in strategy_data.items():
            if df.empty or "Exit_Date" not in df.columns:
                continue
            ts = df.sort_values("Exit_Date").copy()
            ts["Cumulative_PnL"] = ts["PnL_Currency"].cumsum()
            eq = ts["Cumulative_PnL"]
            eq.index = ts["Exit_Date"]
            s_ratio = trade_based_sharpe(ts, annual_factor=252)
            sort_val = sortino_ratio(eq, annual_factor=252)
            cal_val = calmar_ratio(eq)
            ulcer_val = ulcer_index(eq)
            omega_val = omega_ratio(eq)
            base_kelly = kelly_fraction(ts)
            comp_metrics.append({
                "Strategy": label,
                "Sharpe": s_ratio,
                "Sortino": sort_val,
                "Calmar": cal_val,
                "Ulcer": ulcer_val,
                "Omega": omega_val,
                "Kelly": base_kelly
            })
        comp_metrics_df = pd.DataFrame(comp_metrics)
        st.subheader("Advanced Metrics Table")
        format_dict2 = {col: "{:." + str(global_decimals) + "f}" for col in comp_metrics_df.columns if pd.api.types.is_numeric_dtype(comp_metrics_df[col])}
        st.dataframe(comp_metrics_df.style.format(format_dict2))
        for metric in ["Sharpe", "Sortino", "Calmar", "Ulcer", "Omega", "Kelly"]:
            fig = px.bar(comp_metrics_df, x="Strategy", y=metric, template="plotly_dark", title=f"{metric} Comparison")
            st.plotly_chart(fig, use_container_width=True)

    #####################################
    # 15) STRATEGY RANKING & COMPOSITE SCORE TAB
    #####################################
    with tabs[base_index+3]:
        st.header("Strategy Ranking & Composite Score")
        st.markdown("Customize the composite score by selecting metrics and assigning weightage.")
        default_metrics = {
            "Total PnL": 1.0,
            "Win Rate (%)": 1.0,
            "Profit Factor": 1.0,
            "Achieved RR": 1.0,
            "Max Drawdown": -1.0
        }
        selected_metrics = st.multiselect("Select Metrics for Composite Score", list(default_metrics.keys()), default=list(default_metrics.keys()))
        user_weights = {}
        for metric in selected_metrics:
            user_weights[metric] = st.number_input(f"Weight for {metric}", value=float(default_metrics[metric]))
        # We'll reuse the comp_df from the "Strategy Comparison Summary" logic, but let's recalc if needed:
        ranking_df = comp_df.copy()
        for metric in selected_metrics:
            if metric == "Max Drawdown":
                ranking_df[metric + "_rank"] = ranking_df[metric].rank(ascending=True)
            else:
                ranking_df[metric + "_rank"] = ranking_df[metric].rank(ascending=False)
        ranking_df["Composite_Score"] = 0
        for metric in selected_metrics:
            ranking_df["Composite_Score"] += ranking_df[metric + "_rank"] * user_weights[metric]
        ranking_df["Composite_Score"] = ranking_df["Composite_Score"].round(global_decimals)
        ranking_df = ranking_df.sort_values("Composite_Score")
        st.subheader("Composite Ranking Table")
        st.dataframe(ranking_df[["Strategy", "Composite_Score"]])
        fig_rank = px.bar(ranking_df, x="Strategy", y="Composite_Score", template="plotly_dark",
                          title="Composite Score Ranking")
        st.plotly_chart(fig_rank, use_container_width=True)
        categories = [m + "_rank" for m in selected_metrics]
        fig_radar = go.Figure()
        for i, row in ranking_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row[cat] for cat in categories],
                theta=categories,
                fill='toself',
                name=row["Strategy"]
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)),
                                showlegend=True,
                                title="Radar Chart of Metric Ranks")
        st.plotly_chart(fig_radar, use_container_width=True)

    #####################################
    # 16) ENTRY SIZE STATS COMPARISON TAB (Enhanced)
    #####################################
    with tabs[base_index+4]:
        st.header("Entry Size Stats Comparison Across Strategies")
        entry_size_comp = []
        for label, df in strategy_data.items():
            if df.empty or "Entry_Size" not in df.columns:
                continue
            comp = df.groupby("Entry_Size").agg(
                Total_Trades=("PnL_Currency", "count"),
                Total_PnL=("PnL_Currency", "sum"),
                Avg_PnL=("PnL_Currency", "mean"),
                Win_Rate=("PnL_Currency", lambda x: (x > 0).sum() / len(x) * 100)
            ).reset_index()
            if "PnL_per_Lot" in df.columns:
                comp["Avg_PnL_per_Lot"] = df.groupby("Entry_Size")["PnL_per_Lot"].mean().values
            comp["Strategy"] = label
            entry_size_comp.append(comp)
        if entry_size_comp:
            comp_es_df = pd.concat(entry_size_comp, ignore_index=True)
            st.subheader("Comparison Table")
            display_aggrid_table(comp_es_df, key="entry_size_comp_table")
            st.subheader("Bar Charts Comparison")
            fig_es_total = px.bar(comp_es_df, x="Entry_Size", y="Total_PnL", color="Strategy", barmode="group",
                                  template="plotly_dark", title="Total PnL by Entry Size Across Strategies")
            st.plotly_chart(fig_es_total, use_container_width=True)
            if "Avg_PnL_per_Lot" in comp_es_df.columns:
                fig_es_avg = px.bar(comp_es_df, x="Entry_Size", y="Avg_PnL_per_Lot", color="Strategy", barmode="group",
                                    template="plotly_dark", title="Avg PnL per Lot by Entry Size Across Strategies",
                                    hover_data=["Total_PnL", "Avg_PnL_per_Lot", "Win_Rate", "Total_Trades"])
                st.plotly_chart(fig_es_avg, use_container_width=True)
            fig_win_rate = px.bar(comp_es_df, x="Entry_Size", y="Win_Rate", color="Strategy", barmode="group",
                                  template="plotly_dark", title="Win Rate by Entry Size Across Strategies",
                                  hover_data=["Total_PnL", "Avg_PnL_per_Lot", "Win_Rate", "Total_Trades"])
            st.plotly_chart(fig_win_rate, use_container_width=True)
            fig_total_trades = px.bar(comp_es_df, x="Entry_Size", y="Total_Trades", color="Strategy", barmode="group",
                                      template="plotly_dark", title="Total Trades by Entry Size Across Strategies",
                                      hover_data=["Total_PnL", "Avg_PnL_per_Lot", "Win_Rate", "Total_Trades"])
            st.plotly_chart(fig_total_trades, use_container_width=True)

            # NEW FEATURE: Detailed per-lot comparison with benchmark selector
            st.markdown("---")
            st.header("Detailed Outperformance Analysis by Lot Size")

            # Get unique sizes
            unique_sizes = sorted(comp_es_df["Entry_Size"].unique())

            for size in unique_sizes:
                st.markdown(f"### Entry Size: {size}")
                
                # Filter data for this size
                size_data = comp_es_df[comp_es_df["Entry_Size"] == size].copy()
                
                # Strategies available for this size
                avail_strats = size_data["Strategy"].unique().tolist()
                
                # Selector
                bench_strat = st.selectbox(f"Select Benchmark Strategy for Size {size}", 
                                           options=avail_strats, 
                                           key=f"bench_sel_{size}")
                
                # Get Benchmark Stats
                bench_row = size_data[size_data["Strategy"] == bench_strat].iloc[0]
                
                # Safe extraction of metrics
                b_total = bench_row['Total_PnL']
                b_avg = bench_row.get('Avg_PnL_per_Lot', np.nan) # Handle if missing
                b_win = bench_row['Win_Rate']
                b_trades = bench_row['Total_Trades']

                # Display Benchmark Reference
                st.markdown(f"**Benchmark Reference ({bench_strat})**")
                col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                col_b1.metric("Total PnL", f"{b_total:.{global_decimals}f}")
                if not np.isnan(b_avg):
                    col_b2.metric("Avg PnL/Lot", f"{b_avg:.{global_decimals}f}")
                else:
                    col_b2.metric("Avg PnL/Lot", "N/A")
                col_b3.metric("Win Rate", f"{b_win:.{global_decimals}f}%")
                col_b4.metric("Trades", int(b_trades))

                # Filter for outperformers
                # Logic: Total_PnL > Benchmark AND Avg_PnL_per_Lot > Benchmark
                if not np.isnan(b_avg) and "Avg_PnL_per_Lot" in size_data.columns:
                    outperformers = size_data[
                        (size_data["Total_PnL"] > b_total) & 
                        (size_data["Avg_PnL_per_Lot"] > b_avg)
                    ].copy()
                    
                    display_cols = ["Strategy", "Total_PnL", "Avg_PnL_per_Lot", "Win_Rate", "Total_Trades"]
                    # Ensure columns exist
                    display_cols = [c for c in display_cols if c in outperformers.columns]
                    
                    st.markdown(f"**Strategies Outperforming {bench_strat} (Higher Total PnL AND Higher Avg PnL/Lot)**")
                    if not outperformers.empty:
                        # Format for display
                        st.dataframe(outperformers[display_cols].style.format({
                            "Total_PnL": f"{{:.{global_decimals}f}}",
                            "Avg_PnL_per_Lot": f"{{:.{global_decimals}f}}",
                            "Win_Rate": f"{{:.{global_decimals}f}}",
                            "Total_Trades": "{:.0f}"
                        }))
                    else:
                        st.info(f"No strategies outperformed {bench_strat} on both Total PnL and Avg PnL/Lot at this size.")
                else:
                    st.warning("Avg PnL per Lot data missing, cannot perform full comparison.")
                
                st.markdown("---")

        else:
            st.info("No Entry Size data available across strategies.")

    #####################################
    # 17) STRATEGY JUDGEMENT RATIO COMPARISON TAB (New)
    #####################################
    with tabs[base_index+5]:
        st.header("Strategy Judgement Ratio Comparison")
        sr_list = []
        for label, df in strategy_data.items():
            if df.empty or "Exit_Date" not in df.columns:
                continue
            ts = df.sort_values("Exit_Date").copy()
            total_trades = len(ts)
            total_pnl = ts["PnL_Currency"].sum()
            ts["Cumulative_PnL"] = ts["PnL_Currency"].cumsum()
            peak = ts["Cumulative_PnL"].cummax()
            drawdowns = ts["Cumulative_PnL"] - peak
            max_dd = drawdowns.min() if not drawdowns.empty else 0
            if total_trades>0:
                start_date = ts["Exit_Date"].iloc[0]
                end_date = ts["Exit_Date"].iloc[-1]
                total_days = (end_date - start_date).days
                data_years = total_days/365.0 if total_days>0 else 1
            else:
                data_years = 1
            strat_judgment_ratio = (total_pnl/abs(max_dd))/data_years if max_dd<0 else np.nan
            sr_list.append({
                "Strategy": label,
                "Total PnL": total_pnl,
                "Max Drawdown": max_dd,
                "Strategy Judgement Ratio": strat_judgment_ratio
            })
        if sr_list:
            sr_df = pd.DataFrame(sr_list)
            st.subheader("Comparison Table")
            st.dataframe(sr_df.style.format({col: "{:." + str(global_decimals) + "f}" for col in sr_df.select_dtypes(include=[np.number]).columns}))
            fig_sr = px.bar(sr_df, x="Strategy", y="Strategy Judgement Ratio", template="plotly_dark",
                            title="Strategy Judgement Ratio Comparison")
            st.plotly_chart(fig_sr, use_container_width=True)
        else:
            st.info("No Strategy Judgement Ratio data available.")
else:
    st.info("Upload at least 2 strategies to view comparison tabs.")