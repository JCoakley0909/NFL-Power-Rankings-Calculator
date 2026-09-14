import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import nfl_data_py as nfl
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(page_title="NFL Power Rankings Model", layout="wide")

st.title("🏈 7% SOS-Adjusted NFL Power Rankings")
st.markdown("Interactive dashboard evaluating modern NFL team efficiency, explosive rates, and schedule difficulty.")

# -----------------------------------------------------------------------------
# 1. DATA PROCESSING ENGINE (Cached for performance)
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_calculate_rankings(season: int, sos_weight: float = 0.07):
    pbp = nfl.import_pbp_data([season])

    clean_pbp = pbp[
        (pbp['season_type'] == 'REG') &
        (pbp['play_type'].isin(['pass', 'run'])) &
        (pbp['posteam'].notna()) &
        (pbp['defteam'].notna())
    ].copy()

    # Success Definition
    clean_pbp['is_success'] = np.where(
        (clean_pbp['down'] == 1) & (clean_pbp['yards_gained'] >= 0.5 * clean_pbp['ydstogo']), 1,
        np.where(
            (clean_pbp['down'] == 2) & (clean_pbp['yards_gained'] >= 0.7 * clean_pbp['ydstogo']), 1,
            np.where((clean_pbp['down'].isin([3, 4])) & (clean_pbp['yards_gained'] >= 1.0 * clean_pbp['ydstogo']), 1, 0)
        )
    )

    # Explosive Plays
    clean_pbp['is_explosive'] = np.where(
        (clean_pbp['play_type'] == 'run') & (clean_pbp['yards_gained'] >= 10), 1,
        np.where((clean_pbp['play_type'] == 'pass') & (clean_pbp['yards_gained'] >= 20), 1, 0)
    )

    # Offense Stats
    off_base = clean_pbp.groupby('posteam').agg(
        off_epa=('epa', 'mean'),
        off_success=('is_success', 'mean'),
        off_explosive=('is_explosive', 'mean')
    ).reset_index()

    pass_plays = clean_pbp[clean_pbp['pass'] == 1]
    off_nya = pass_plays.groupby('posteam').agg(
        pass_yards=('passing_yards', 'sum'),
        pass_attempts=('pass_attempt', 'sum'),
        sacks=('sack', 'sum')
    ).reset_index()
    off_nya['off_nya'] = (off_nya['pass_yards']) / (off_nya['pass_attempts'] + off_nya['sacks'])

    rz_plays = clean_pbp[(clean_pbp['yardline_100'] <= 20)]
    rz_drives = rz_plays.groupby(['posteam', 'drive']).agg(
        td_scored=('touchdown', lambda x: 1 if (x == 1).any() else 0)
    ).reset_index()
    off_rz = rz_drives.groupby('posteam')['td_scored'].mean().reset_index().rename(columns={'td_scored': 'off_rz_td_pct'})

    off_to = clean_pbp.groupby('posteam').agg(
        off_turnover_rate=('fumble_lost', lambda x: (x.sum() + clean_pbp.loc[x.index, 'interception'].sum()) / len(x))
    ).reset_index()

    offense_df = off_base.merge(off_nya[['posteam', 'off_nya']], on='posteam')
    offense_df = offense_df.merge(off_rz, on='posteam')
    offense_df = offense_df.merge(off_to, on='posteam')

    # Defense Stats
    def_base = clean_pbp.groupby('defteam').agg(
        def_epa=('epa', 'mean'),
        def_success=('is_success', 'mean'),
        def_explosive=('is_explosive', 'mean')
    ).reset_index()

    def_press = pass_plays.groupby('defteam').agg(
        def_pressure_rate=('qb_hit', lambda x: (x.sum() + pass_plays.loc[x.index, 'sack'].sum()) / len(x))
    ).reset_index()

    clean_pbp['is_tfl'] = np.where((clean_pbp['play_type'] == 'run') & (clean_pbp['yards_gained'] < 0), 1, 0)
    clean_pbp['is_pbu'] = clean_pbp['pass_defense_1_player_id'].notna().astype(int)

    def_havoc = clean_pbp.groupby('defteam').agg(
        def_havoc_rate=('is_tfl', lambda x: (
            x.sum() + 
            clean_pbp.loc[x.index, 'sack'].sum() + 
            clean_pbp.loc[x.index, 'fumble_forced'].sum() + 
            clean_pbp.loc[x.index, 'is_pbu'].sum()
        ) / len(x))
    ).reset_index()

    defense_df = def_base.merge(def_press, on='defteam')
    defense_df = defense_df.merge(def_havoc, on='defteam')

    df = offense_df.merge(defense_df, left_on='posteam', right_on='defteam').drop(columns=['defteam']).rename(columns={'posteam': 'Team'})

    # Metric Ranks
    df['r_off_epa'] = df['off_epa'].rank(ascending=False)
    df['r_off_success'] = df['off_success'].rank(ascending=False)
    df['r_off_explosive'] = df['off_explosive'].rank(ascending=False)
    df['r_off_nya'] = df['off_nya'].rank(ascending=False)
    df['r_off_rz'] = df['off_rz_td_pct'].rank(ascending=False)
    df['r_off_to'] = df['off_turnover_rate'].rank(ascending=True)

    df['r_def_epa'] = df['def_epa'].rank(ascending=True)
    df['r_def_success'] = df['def_success'].rank(ascending=True)
    df['r_def_pressure'] = df['def_pressure_rate'].rank(ascending=False)
    df['r_def_explosive'] = df['def_explosive'].rank(ascending=True)
    df['r_def_havoc'] = df['def_havoc_rate'].rank(ascending=False)

    df['Offense_Score'] = (
        (df['r_off_epa'] * 0.25) + (df['r_off_success'] * 0.20) +
        (df['r_off_explosive'] * 0.15) + (df['r_off_nya'] * 0.15) +
        (df['r_off_rz'] * 0.15) + (df['r_off_to'] * 0.10)
    )

    df['Defense_Score'] = (
        (df['r_def_epa'] * 0.25) + (df['r_def_success'] * 0.25) +
        (df['r_def_pressure'] * 0.20) + (df['r_def_explosive'] * 0.15) +
        (df['r_def_havoc'] * 0.15)
    )

    df['Raw_Score'] = (df['Offense_Score'] * 0.55) + (df['Defense_Score'] * 0.45)

    # SOS Calculation
    games = clean_pbp[['game_id', 'home_team', 'away_team']].drop_duplicates()
    sos_dict = {}
    raw_score_map = dict(zip(df['Team'], df['Raw_Score']))

    for team in df['Team']:
        home_opps = games[games['home_team'] == team]['away_team'].tolist()
        away_opps = games[games['away_team'] == team]['home_team'].tolist()
        opponents = home_opps + away_opps
        opp_scores = [raw_score_map[opp] for opp in opponents if opp in raw_score_map]
        sos_dict[team] = np.mean(opp_scores) if opp_scores else 16.5

    df['Avg_Opponent_Raw_Score'] = df['Team'].map(sos_dict)
    df['SOS_Rank'] = df['Avg_Opponent_Raw_Score'].rank(ascending=True)

    # 7% SOS Final Adjustment
    df['Final_Power_Score'] = (df['Raw_Score'] * (1.0 - sos_weight)) + (df['SOS_Rank'] * sos_weight)
    df['Power_Rank'] = df['Final_Power_Score'].rank(ascending=True).astype(int)

    return df.sort_values('Power_Rank').reset_index(drop=True)

# -----------------------------------------------------------------------------
# 2. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("Dashboard Settings")
selected_season = st.sidebar.selectbox("Select Season", options=[2025, 2024, 2023, 2022], index=0)
sos_setting = st.sidebar.slider("Strength of Schedule Weight (%)", min_value=0.0, max_value=0.20, value=0.07, step=0.01)

# Fetch data based on sidebar controls
df_rankings = load_and_calculate_rankings(selected_season, sos_weight=sos_setting)

# -----------------------------------------------------------------------------
# 3. TOP SUMMARY METRIC CARDS
# -----------------------------------------------------------------------------
top_team = df_rankings.iloc[0]['Team']
top_offense = df_rankings.sort_values('Offense_Score').iloc[0]['Team']
top_defense = df_rankings.sort_values('Defense_Score').iloc[0]['Team']

col1, col2, col3 = st.columns(3)
col1.metric(label="🏆 #1 Ranked Team Overall", value=top_team)
col2.metric(label="💥 #1 Offense Unit", value=top_offense)
col3.metric(label="🔒 #1 Defense Unit", value=top_defense)

st.markdown("---")

# -----------------------------------------------------------------------------
# 4. PLOTLY INTERACTIVE CHARTS
# -----------------------------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Offense vs. Defense Rank Tiering")
    fig_scatter = px.scatter(
        df_rankings,
        x="Offense_Score",
        y="Defense_Score",
        text="Team",
        size="Final_Power_Score",
        color="Power_Rank",
        color_continuous_scale="Viridis_r",
        hover_data=["SOS_Rank"],
        labels={"Offense_Score": "Offense Rank Score (Lower = Better)", "Defense_Score": "Defense Rank Score (Lower = Better)"}
    )
    fig_scatter.update_traces(textposition='top center', marker=dict(sizeref=0.5))
    fig_scatter.update_xaxes(autorange="reverse")
    fig_scatter.update_yaxes(autorange="reverse")
    st.plotly_chart(fig_scatter, use_container_width=True)

with chart_col2:
    st.subheader("Top 10 Final Power Scores")
    top_10 = df_rankings.head(10)
    fig_bar = px.bar(
        top_10,
        x="Team",
        y="Final_Power_Score",
        color="Final_Power_Score",
        color_continuous_scale="bluered",
        labels={"Final_Power_Score": "Power Score (Lower = Better)"},
        text_auto=".2f"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------------------------------------------------------
# 5. DATA TABLE VIEW
# -----------------------------------------------------------------------------
st.subheader(f"Full NFL Standings & Breakdown ({selected_season})")

st.dataframe(
    df_rankings[['Power_Rank', 'Team', 'Final_Power_Score', 'Raw_Score', 'SOS_Rank', 'Offense_Score', 'Defense_Score']],
    column_config={
        "Power_Rank": st.column_config.NumberColumn("Rank", format="%d"),
        "Final_Power_Score": st.column_config.NumberColumn("Final Score", format="%.2f"),
        "Raw_Score": st.column_config.NumberColumn("Performance Score", format="%.2f"),
        "SOS_Rank": st.column_config.NumberColumn("SOS Rank", format="%d"),
        "Offense_Score": st.column_config.NumberColumn("Offense Score", format="%.2f"),
        "Defense_Score": st.column_config.NumberColumn("Defense Score", format="%.2f"),
    },
    hide_index=True,
    use_container_width=True
)