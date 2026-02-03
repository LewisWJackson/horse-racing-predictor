"""
Horse Racing Prediction Dashboard
=================================
Complete betting analysis and prediction dashboard.

Features:
- Race predictions with win/place/each-way recommendations
- Value bet finder (predicted probability vs odds)
- Accumulator builder with expected value calculation
- ELO rankings for horses, jockeys, trainers
- Form analysis and historical performance
- Bankroll management and staking suggestions

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

st.set_page_config(
    page_title="Horse Racing Predictor",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a472a;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .bet-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin-bottom: 0.5rem;
    }
    .value-bet {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
    }
    .warning-bet {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
    }
    .danger-bet {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def load_upcoming_races():
    """Load upcoming races for next 7 days."""
    upcoming_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'upcoming_races.csv')
    try:
        df = pd.read_csv(upcoming_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception:
        # Generate from ELO data if file not found (e.g. first deploy)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from fetch_upcoming_races import get_upcoming_races
        return get_upcoming_races(use_cache=False)


@st.cache_resource
def load_model():
    """Load the trained model."""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except:
        return None


@st.cache_resource
def load_elo_ratings():
    """Load ELO ratings from trained model."""
    elo_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'elo_ratings.pkl')
    try:
        with open(elo_path, 'rb') as f:
            elo_data = pickle.load(f)
        return elo_data
    except:
        return {'horse_elo': {}, 'jockey_elo': {}, 'trainer_elo': {}}


def parse_odds(sp: str) -> float:
    """Parse starting price odds."""
    if pd.isna(sp):
        return 20.0
    sp_str = str(sp).strip().upper().replace('F', '').replace('J', '').replace('C', '')
    if 'EVENS' in sp_str or 'EVS' in sp_str:
        return 2.0
    try:
        if '/' in sp_str:
            parts = sp_str.split('/')
            return (float(parts[0]) / float(parts[1])) + 1
        else:
            return float(sp_str)
    except:
        return 20.0


def calculate_implied_probability(decimal_odds: float) -> float:
    """Calculate implied probability from decimal odds."""
    if decimal_odds <= 1:
        return 1.0
    return 1 / decimal_odds


def calculate_value(predicted_prob: float, decimal_odds: float) -> float:
    """Calculate betting value (edge)."""
    implied_prob = calculate_implied_probability(decimal_odds)
    return predicted_prob - implied_prob


def kelly_criterion(prob: float, odds: float, fraction: float = 0.25) -> float:
    """Calculate Kelly Criterion stake (fractional for safety)."""
    if odds <= 1 or prob <= 0 or prob >= 1:
        return 0
    b = odds - 1
    q = 1 - prob
    kelly = (b * prob - q) / b
    return max(0, kelly * fraction)


def calculate_accumulator_odds(selections: list) -> float:
    """Calculate combined odds for accumulator."""
    if not selections:
        return 1.0
    combined = 1.0
    for odds in selections:
        combined *= odds
    return combined


def decimal_to_fractional(decimal_odds: float) -> str:
    """Convert decimal odds to fractional format (e.g., 3.5 -> 5/2)."""
    if decimal_odds <= 1:
        return "1/1"

    fraction = decimal_odds - 1

    # Common fractional odds lookup
    common_fractions = {
        0.1: "1/10", 0.11: "1/9", 0.125: "1/8", 0.14: "1/7", 0.17: "1/6",
        0.2: "1/5", 0.22: "2/9", 0.25: "1/4", 0.29: "2/7", 0.3: "3/10",
        0.33: "1/3", 0.36: "4/11", 0.4: "2/5", 0.44: "4/9", 0.45: "9/20",
        0.5: "1/2", 0.53: "8/15", 0.55: "11/20", 0.57: "4/7", 0.6: "3/5",
        0.62: "8/13", 0.64: "4/11", 0.67: "2/3", 0.7: "7/10", 0.73: "8/11",
        0.75: "3/4", 0.8: "4/5", 0.83: "5/6", 0.91: "10/11", 0.95: "20/21",
        1.0: "1/1", 1.1: "11/10", 1.2: "6/5", 1.25: "5/4", 1.3: "13/10",
        1.33: "4/3", 1.36: "15/11", 1.4: "7/5", 1.5: "3/2", 1.6: "8/5",
        1.67: "5/3", 1.73: "19/11", 1.75: "7/4", 1.8: "9/5", 1.83: "11/6",
        2.0: "2/1", 2.1: "21/10", 2.2: "11/5", 2.25: "9/4", 2.38: "19/8",
        2.5: "5/2", 2.75: "11/4", 3.0: "3/1", 3.5: "7/2", 4.0: "4/1",
        4.5: "9/2", 5.0: "5/1", 5.5: "11/2", 6.0: "6/1", 6.5: "13/2",
        7.0: "7/1", 7.5: "15/2", 8.0: "8/1", 9.0: "9/1", 10.0: "10/1",
        11.0: "11/1", 12.0: "12/1", 14.0: "14/1", 16.0: "16/1", 20.0: "20/1",
        25.0: "25/1", 33.0: "33/1", 40.0: "40/1", 50.0: "50/1", 66.0: "66/1",
        80.0: "80/1", 100.0: "100/1", 150.0: "150/1", 200.0: "200/1"
    }

    # Find closest match
    closest = min(common_fractions.keys(), key=lambda x: abs(x - fraction))
    if abs(closest - fraction) < 0.05:
        return common_fractions[closest]

    # Calculate custom fraction
    if fraction >= 1:
        # Odds-on or greater
        numerator = round(fraction)
        return f"{numerator}/1"
    else:
        # Odds-against
        denominator = round(1 / fraction)
        return f"1/{denominator}"


def calculate_accumulator_probability(probabilities: list) -> float:
    """Calculate combined probability for accumulator."""
    if not probabilities:
        return 0.0
    combined = 1.0
    for prob in probabilities:
        combined *= prob
    return combined


@st.cache_data(ttl=1800)  # Cache for 30 mins
def generate_todays_picks(upcoming_df: pd.DataFrame, elo_data: dict) -> dict:
    """
    Generate best picks from upcoming races in next 7 days.
    Uses ELO ratings and odds analysis to find value bets.
    """
    horse_elo = elo_data.get('horse_elo', {})
    jockey_elo = elo_data.get('jockey_elo', {})
    trainer_elo = elo_data.get('trainer_elo', {})

    all_picks = []

    # Group by race (date + course + time)
    for (date, course, time), race in upcoming_df.groupby(['date', 'course', 'time']):
        if len(race) < 3:
            continue

        race_info = {
            'date': date,
            'day': race['day'].iloc[0] if 'day' in race.columns else '',
            'course': course,
            'race_name': race['race_name'].iloc[0] if 'race_name' in race.columns else f"{course} Race",
            'time': time,
            'distance': race['distance'].iloc[0] if 'distance' in race.columns else '',
            'going': race['going'].iloc[0] if 'going' in race.columns else ''
        }

        race_id = f"{date}_{course}_{time}"

        # First pass: collect ELO data for all runners in the race
        race_runners = []
        for _, row in race.iterrows():
            odds = float(row.get('odds', 20))
            if odds < 1.1:
                odds = 1.5

            h_elo = row.get('horse_elo', horse_elo.get(row['horse'], 1500))
            j_elo = row.get('jockey_elo', jockey_elo.get(row.get('jockey', ''), 1500))
            t_elo = row.get('trainer_elo', trainer_elo.get(row.get('trainer', ''), 1500))
            composite_elo = h_elo * 0.5 + j_elo * 0.3 + t_elo * 0.2

            race_runners.append({
                'row': row,
                'odds': odds,
                'h_elo': h_elo,
                'j_elo': j_elo,
                't_elo': t_elo,
                'composite_elo': composite_elo
            })

        if len(race_runners) < 2:
            continue

        # Second pass: calculate normalised probabilities across the field
        import numpy as np_local
        elos = np_local.array([r['composite_elo'] for r in race_runners])
        elo_centered = elos - np_local.mean(elos)
        exp_elos = np_local.exp(elo_centered / 100)
        model_probs = exp_elos / exp_elos.sum()

        for i, rr in enumerate(race_runners):
            row = rr['row']
            odds = rr['odds']
            implied_prob = 1 / odds
            model_prob = float(model_probs[i])
            top3_prob = min(0.92, model_prob * 2.5)
            value_edge = model_prob - implied_prob

            if value_edge > 0.02:  # Only picks with 2%+ edge
                all_picks.append({
                    'race_id': race_id,
                    'date': race_info['date'],
                    'day': race_info['day'],
                    'course': race_info['course'],
                    'race_name': race_info['race_name'],
                    'time': race_info['time'],
                    'distance': race_info['distance'],
                    'going': race_info['going'],
                    'horse': row['horse'],
                    'jockey': row.get('jockey', 'Unknown'),
                    'trainer': row.get('trainer', 'Unknown'),
                    'odds': odds,
                    'implied_prob': implied_prob,
                    'model_prob': model_prob,
                    'top3_prob': top3_prob,
                    'value_edge': value_edge,
                    'horse_elo': rr['h_elo'],
                    'jockey_elo': rr['j_elo'],
                    'trainer_elo': rr['t_elo'],
                    'composite_elo': rr['composite_elo'],
                    'kelly_stake': kelly_criterion(model_prob, odds)
                })

    # Sort by value edge
    all_picks.sort(key=lambda x: x['value_edge'], reverse=True)

    # Generate accumulator suggestions
    accumulators = []
    if len(all_picks) >= 3:
        # Group picks by race to avoid same-race selections
        race_groups = {}
        for pick in all_picks[:30]:  # Top 30 picks
            rid = pick['race_id']
            if rid not in race_groups:
                race_groups[rid] = []
            race_groups[rid].append(pick)

        # Get best pick from each race
        best_per_race = []
        for rid, picks in race_groups.items():
            best_per_race.append(max(picks, key=lambda x: x['value_edge']))

        # Generate 3-fold, 4-fold accumulators
        from itertools import combinations
        for size in [3, 4]:
            if len(best_per_race) >= size:
                best_combos = []
                for combo in combinations(best_per_race[:10], size):
                    combined_odds = 1
                    combined_prob = 1
                    for pick in combo:
                        combined_odds *= pick['odds']
                        combined_prob *= pick['model_prob']

                    implied = 1 / combined_odds
                    edge = combined_prob - implied
                    ev = combined_prob * combined_odds - 1

                    if edge > 0:
                        best_combos.append({
                            'selections': list(combo),
                            'combined_odds': combined_odds,
                            'combined_prob': combined_prob,
                            'value_edge': edge,
                            'expected_value': ev,
                            'size': size
                        })

                best_combos.sort(key=lambda x: x['expected_value'], reverse=True)
                accumulators.extend(best_combos[:3])

    return {
        'win_picks': all_picks[:15],
        'place_picks': sorted(all_picks, key=lambda x: x['top3_prob'], reverse=True)[:10],
        'value_picks': [p for p in all_picks if p['value_edge'] > 0.08][:10],
        'accumulators': accumulators[:6]
    }


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.markdown("# 🏇 Horse Racing Predictor")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🎯 Race Predictions", "💰 Value Bets", "🎰 Accumulator Builder",
     "📈 ELO Rankings", "📋 Bet Tracker", "⚙️ Settings"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Performance")
st.sidebar.metric("Winner Accuracy", "35.2%", "+25.2pp vs random")
st.sidebar.metric("Top 3 Accuracy", "68.9%", "+35.6pp vs random")
st.sidebar.metric("AUC-ROC", "0.799", "Excellent")

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Stats")
model_data = load_model()
if model_data:
    st.sidebar.success("✅ Model Loaded")
else:
    st.sidebar.error("❌ Model Not Found")


# ============================================================================
# MAIN PAGES
# ============================================================================

if page == "📊 Dashboard":
    st.markdown('<h1 class="main-header">🏇 TODAY\'S BEST BETS</h1>', unsafe_allow_html=True)

    # Load upcoming races and generate picks
    upcoming_df = load_upcoming_races()
    elo_data = load_elo_ratings()
    picks = generate_todays_picks(upcoming_df, elo_data)

    # Date range info
    today = datetime.now().date()
    st.info(f"📅 Showing races from **{today.strftime('%A %d %B')}** to **{(today + timedelta(days=6)).strftime('%A %d %B %Y')}** ({len(upcoming_df.groupby(['date', 'course', 'time']))} races across {upcoming_df['course'].nunique()} courses)")

    # Summary metrics at top
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Value Bets Found", len(picks['win_picks']))
    with col2:
        st.metric("⭐ High Confidence", len(picks['value_picks']))
    with col3:
        st.metric("🎰 Accumulators", len(picks['accumulators']))
    with col4:
        if picks['win_picks']:
            best_edge = picks['win_picks'][0]['value_edge'] * 100
            st.metric("📈 Best Edge", f"+{best_edge:.1f}%")

    st.markdown("---")

    # ===== TOP PICK OF THE DAY =====
    if picks['win_picks']:
        top_pick = picks['win_picks'][0]
        st.markdown("## 🏆 TOP PICK OF THE DAY")

        col1, col2 = st.columns([2, 1])
        with col1:
            pick_day = top_pick.get('day', '')
            pick_date = pd.to_datetime(top_pick['date']).strftime('%d %B') if top_pick.get('date') else ''
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%); padding: 25px; border-radius: 15px; color: white;">
                <h2 style="margin:0; color: #ffd700;">{top_pick['horse']}</h2>
                <p style="font-size: 1.2rem; margin: 10px 0;">📅 {pick_day} {pick_date} | ⏰ {top_pick['time']} | 📍 {top_pick['course']}</p>
                <p style="margin: 5px 0;">{top_pick['race_name']} | {top_pick.get('distance', '')} | {top_pick.get('going', '')}</p>
                <p style="margin: 5px 0;">🏇 Jockey: {top_pick['jockey']} | 🎓 Trainer: {top_pick['trainer']}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.metric("Odds", decimal_to_fractional(top_pick['odds']))
            st.metric("Win Probability", f"{top_pick['model_prob']*100:.0f}%")
            st.metric("Value Edge", f"+{top_pick['value_edge']*100:.1f}%")
            st.metric("Suggested Stake", f"{top_pick['kelly_stake']*100:.1f}% of bankroll")

    st.markdown("---")

    # ===== BEST WIN BETS =====
    st.markdown("## 🎯 BEST WIN BETS")
    st.caption("Highest value edge - model predicts higher probability than odds suggest")

    if picks['win_picks']:
        for i, pick in enumerate(picks['win_picks'][:5], 1):
            col1, col2, col3, col4, col5, col6 = st.columns([0.5, 2.5, 1.5, 1, 1, 1])

            with col1:
                st.markdown(f"**#{i}**")
            with col2:
                pick_day = pick.get('day', '')[:3]  # Mon, Tue, etc
                st.markdown(f"**{pick['horse']}**")
                st.caption(f"{pick_day} {pick['time']} - {pick['course']}")
            with col3:
                st.markdown(f"🏇 {pick['jockey'][:20]}")
            with col4:
                st.markdown(f"**{decimal_to_fractional(pick['odds'])}**")
            with col5:
                st.markdown(f"**{pick['model_prob']*100:.0f}%** win")
            with col6:
                edge_color = "green" if pick['value_edge'] > 0.1 else "orange"
                st.markdown(f"<span style='color:{edge_color}; font-weight:bold;'>+{pick['value_edge']*100:.1f}%</span>", unsafe_allow_html=True)

        st.markdown("---")

    # ===== BEST PLACE BETS (TOP 3) =====
    st.markdown("## 🥉 BEST PLACE BETS (Top 3)")
    st.caption("Highest probability of finishing in the top 3")

    if picks['place_picks']:
        for i, pick in enumerate(picks['place_picks'][:5], 1):
            col1, col2, col3, col4, col5, col6 = st.columns([0.5, 2.5, 1.5, 1, 1, 1])

            with col1:
                st.markdown(f"**#{i}**")
            with col2:
                pick_day = pick.get('day', '')[:3]
                st.markdown(f"**{pick['horse']}**")
                st.caption(f"{pick_day} {pick['time']} - {pick['course']}")
            with col3:
                st.markdown(f"🏇 {pick['jockey'][:20]}")
            with col4:
                st.markdown(f"**{decimal_to_fractional(pick['odds'])}**")
            with col5:
                st.markdown(f"**{pick['top3_prob']*100:.0f}%** place")
            with col6:
                st.markdown(f"Win: {pick['model_prob']*100:.0f}%")

        st.markdown("---")

    # ===== RECOMMENDED ACCUMULATORS =====
    st.markdown("## 🎰 RECOMMENDED ACCUMULATORS")
    st.caption("Best value accumulator combinations - different races only")

    if picks['accumulators']:
        for i, acca in enumerate(picks['accumulators'][:4], 1):
            with st.expander(f"📋 {acca['size']}-FOLD ACCUMULATOR #{i} - Odds: {decimal_to_fractional(acca['combined_odds'])} | EV: +{acca['expected_value']*100:.1f}%", expanded=(i==1)):
                st.markdown(f"**Combined Win Probability:** {acca['combined_prob']*100:.2f}%")
                st.markdown(f"**Value Edge:** +{acca['value_edge']*100:.2f}%")
                st.markdown(f"**Expected Return per £1:** £{acca['expected_value']+1:.2f}")

                st.markdown("### Selections:")
                for j, sel in enumerate(acca['selections'], 1):
                    st.markdown(f"""
                    **{j}. {sel['horse']}** @ {decimal_to_fractional(sel['odds'])}
                    - {sel['time']} {sel['course']}
                    - Jockey: {sel['jockey']} | Win Prob: {sel['model_prob']*100:.0f}%
                    """)

                # Returns calculator
                stake = st.number_input(f"Stake (£)", min_value=1.0, value=5.0, key=f"acca_stake_{i}")
                potential_return = stake * acca['combined_odds']
                st.success(f"💰 £{stake:.2f} returns **£{potential_return:.2f}** (Profit: £{potential_return-stake:.2f})")
    else:
        st.info("No accumulator suggestions available for recent races.")

    st.markdown("---")

    # ===== QUICK STATS =====
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Model Performance")
        st.markdown("""
        | Metric | Value |
        |--------|-------|
        | Winner Accuracy | 35.2% |
        | Top 3 Accuracy | 68.9% |
        | AUC-ROC | 0.799 |
        | Edge vs Random | +25pp |
        """)

    with col2:
        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        - **Value Edge > 10%**: Strong bet
        - **Value Edge 5-10%**: Moderate bet
        - **Top 3 Prob > 60%**: Good for place/each-way
        - **Kelly Stake**: Never bet more than suggested
        """)


elif page == "🎯 Race Predictions":
    st.markdown('<h1 class="main-header">🎯 Race-by-Race Picks</h1>', unsafe_allow_html=True)

    # Load upcoming races for next 7 days
    upcoming_df = load_upcoming_races()
    elo_data = load_elo_ratings()
    horse_elo = elo_data.get('horse_elo', {})
    jockey_elo = elo_data.get('jockey_elo', {})
    trainer_elo = elo_data.get('trainer_elo', {})

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        # Get dates from upcoming races
        upcoming_dates = sorted(upcoming_df['date'].dt.date.unique())
        date_options = [f"{d.strftime('%A %d %B')}" for d in upcoming_dates]
        date_map = dict(zip(date_options, upcoming_dates))
        selected_date_str = st.selectbox("Select Date", date_options)
        selected_date = date_map[selected_date_str]

    with col2:
        day_races = upcoming_df[upcoming_df['date'].dt.date == selected_date]
        courses = sorted(day_races['course'].unique())
        selected_course = st.selectbox("Select Course", courses if len(courses) > 0 else ["No races"])

    with col3:
        if selected_course != "No races":
            course_races = day_races[day_races['course'] == selected_course]
            race_times = sorted(course_races['time'].unique())
            selected_race = st.selectbox("Select Race Time", race_times)

    st.markdown("---")

    if selected_course != "No races":
        # Get race data
        race_data = day_races[(day_races['course'] == selected_course) & (day_races['time'] == selected_race)]

        if len(race_data) > 0:
            race_name = race_data['race_name'].iloc[0] if 'race_name' in race_data.columns else f"{selected_course} Race"
            distance = race_data['distance'].iloc[0] if 'distance' in race_data.columns else 'N/A'
            going = race_data['going'].iloc[0] if 'going' in race_data.columns else 'N/A'
            race_type = race_data['race_type'].iloc[0] if 'race_type' in race_data.columns else 'N/A'

            st.subheader(f"📍 {race_name}")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Course", selected_course)
            col2.metric("Distance", distance)
            col3.metric("Going", going)
            col4.metric("Runners", len(race_data))

            st.markdown("---")

            # Display runners with predictions
            st.subheader("🏇 Runners & Model Predictions")

            # Process runners with ELO-based predictions (normalised across the field)
            race_elos = []
            race_odds_list = []
            for _, row in race_data.iterrows():
                odds = float(row.get('odds', 20))
                if odds < 1.1:
                    odds = 1.5
                h_elo = horse_elo.get(row['horse'], 1500)
                j_elo = jockey_elo.get(row.get('jockey', ''), 1500)
                t_elo = trainer_elo.get(row.get('trainer', ''), 1500)
                composite_elo = h_elo * 0.5 + j_elo * 0.3 + t_elo * 0.2
                race_elos.append(composite_elo)
                race_odds_list.append(odds)

            # Normalised probabilities via softmax on ELO
            elos_arr = np.array(race_elos)
            elo_centered = elos_arr - np.mean(elos_arr)
            exp_elos = np.exp(elo_centered / 100)
            model_probs = exp_elos / exp_elos.sum()

            runners = []
            for idx, (_, row) in enumerate(race_data.iterrows()):
                odds = race_odds_list[idx]
                implied_prob = calculate_implied_probability(odds)
                model_prob = float(model_probs[idx])
                top3_prob = min(0.92, model_prob * 2.5)
                value_edge = model_prob - implied_prob
                composite_elo = race_elos[idx]
                kelly = kelly_criterion(model_prob, odds)

                # Determine recommendation
                if value_edge > 0.10:
                    rec = '🔥 STRONG VALUE'
                elif value_edge > 0.05:
                    rec = '✅ VALUE BET'
                elif value_edge > 0:
                    rec = '⚠️ MARGINAL'
                else:
                    rec = '❌ NO VALUE'

                runners.append({
                    'Horse': row['horse'],
                    'Jockey': row.get('jockey', 'Unknown'),
                    'Odds': decimal_to_fractional(odds),
                    'Win %': f"{model_prob*100:.0f}%",
                    'Top 3 %': f"{top3_prob*100:.0f}%",
                    'Edge': f"{value_edge*100:+.1f}%",
                    'ELO': int(composite_elo),
                    'Kelly %': f"{kelly*100:.1f}%",
                    'Verdict': rec,
                    '_model_prob': model_prob,
                    '_value_edge': value_edge,
                    '_odds': odds
                })

            # Sort by model probability
            runners.sort(key=lambda x: x['_model_prob'], reverse=True)

            # Add rank
            for i, r in enumerate(runners, 1):
                r['Rank'] = i

            runners_df = pd.DataFrame(runners)
            display_cols = ['Rank', 'Horse', 'Jockey', 'Odds', 'Win %', 'Top 3 %', 'Edge', 'ELO', 'Kelly %', 'Verdict']

            # Color code the dataframe
            def highlight_value(val):
                if '🔥' in str(val) or 'STRONG' in str(val):
                    return 'background-color: #198754; color: white;'
                elif '✅' in str(val):
                    return 'background-color: #d4edda'
                elif '❌' in str(val):
                    return 'background-color: #f8d7da'
                return ''

            st.dataframe(
                runners_df[display_cols].style.applymap(highlight_value, subset=['Verdict']),
                use_container_width=True,
                hide_index=True
            )

            # Specific betting recommendations for THIS race
            st.markdown("---")
            st.subheader("💰 RECOMMENDED BETS FOR THIS RACE")

            value_runners = [r for r in runners if r['_value_edge'] > 0.03]

            if value_runners:
                col1, col2, col3 = st.columns(3)

                # Best WIN bet
                best_win = max(runners, key=lambda x: x['_value_edge'])
                with col1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #198754 0%, #20c997 100%); padding: 20px; border-radius: 10px; color: white;">
                        <h4>🎯 WIN BET</h4>
                        <h2>{best_win['Horse']}</h2>
                        <p><strong>{best_win['Odds']}</strong> | Win: {best_win['Win %']}</p>
                        <p>Edge: {best_win['Edge']} | Stake: {best_win['Kelly %']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Best EACH-WAY (needs good Top 3 % and decent odds)
                ew_candidates = [r for r in runners if r['_odds'] >= 5 and float(r['Top 3 %'].replace('%','')) > 50]
                with col2:
                    if ew_candidates:
                        best_ew = max(ew_candidates, key=lambda x: float(x['Top 3 %'].replace('%','')))
                        st.markdown(f"""
                        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #0d6efd;">
                            <h4>📊 EACH-WAY</h4>
                            <h2>{best_ew['Horse']}</h2>
                            <p><strong>{best_ew['Odds']}</strong> | Top 3: {best_ew['Top 3 %']}</p>
                            <p>Win: {best_ew['Win %']} | Good E/W value at this price</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #6c757d;">
                            <h4>📊 EACH-WAY</h4>
                            <p>No suitable E/W bets</p>
                            <p>Need 4/1+ odds with good place chance</p>
                        </div>
                        """, unsafe_allow_html=True)

                # Best PLACE bet (highest Top 3 %)
                best_place = max(runners, key=lambda x: float(x['Top 3 %'].replace('%','')))
                with col3:
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #ffc107;">
                        <h4>🥉 PLACE BET</h4>
                        <h2>{best_place['Horse']}</h2>
                        <p><strong>{best_place['Odds']}</strong> | Top 3: {best_place['Top 3 %']}</p>
                        <p>Safest option for this race</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No clear value bets in this race. Consider skipping or betting very small stakes.")


elif page == "💰 Value Bets":
    st.markdown('<h1 class="main-header">💰 Value Bet Finder</h1>', unsafe_allow_html=True)

    # Load upcoming races
    upcoming_df = load_upcoming_races()
    elo_data = load_elo_ratings()
    picks = generate_todays_picks(upcoming_df, elo_data)

    today = datetime.now().date()
    st.info(f"📅 Analyzing **{len(upcoming_df.groupby(['date', 'course', 'time']))}** upcoming races from **{today.strftime('%d %B')}** to **{(today + timedelta(days=6)).strftime('%d %B %Y')}**")

    st.markdown("---")

    # Value threshold selector
    col1, col2, col3 = st.columns(3)

    with col1:
        min_value = st.slider("Minimum Value Edge (%)", 0, 30, 5)

    with col2:
        min_prob = st.slider("Minimum Win Probability (%)", 5, 50, 15)

    with col3:
        bet_type = st.selectbox("Bet Type Filter", ["All", "Win Only", "Place (Top 3)", "Each-Way"])

    st.markdown("---")

    # Real value bets from model
    st.subheader("🎯 RECOMMENDED VALUE BETS")

    # Filter picks based on criteria
    filtered_picks = [p for p in picks['win_picks']
                      if p['value_edge'] * 100 >= min_value
                      and p['model_prob'] * 100 >= min_prob]

    if filtered_picks:
        for i, pick in enumerate(filtered_picks[:15], 1):
            rating = '⭐⭐⭐' if pick['value_edge'] > 0.12 else '⭐⭐' if pick['value_edge'] > 0.08 else '⭐'

            with st.container():
                col1, col2, col3, col4, col5, col6 = st.columns([2.5, 1.5, 1, 1, 1, 1])

                with col1:
                    pick_day = pick.get('day', '')[:3]
                    st.markdown(f"**{i}. {pick['horse']}**")
                    st.caption(f"{pick_day} {pick['time']} - {pick['course']} ({pick.get('distance', '')})")

                with col2:
                    st.markdown(f"**{decimal_to_fractional(pick['odds'])}**")
                    st.caption(f"Jockey: {pick['jockey'][:15]}")

                with col3:
                    st.metric("Win %", f"{pick['model_prob']*100:.0f}%")

                with col4:
                    st.metric("Top 3 %", f"{pick['top3_prob']*100:.0f}%")

                with col5:
                    st.metric("Edge", f"+{pick['value_edge']*100:.1f}%", delta=rating)

                with col6:
                    st.metric("Stake", f"{pick['kelly_stake']*100:.1f}%")

            st.markdown("---")

        # Summary
        st.success(f"✅ Found **{len(filtered_picks)}** value bets matching your criteria")

        # Quick bet slip
        st.subheader("📝 Quick Bet Slip")
        total_stake = 0
        total_ev = 0
        for pick in filtered_picks[:5]:
            stake = pick['kelly_stake'] * 100  # As percentage
            ev = pick['model_prob'] * pick['odds'] - 1
            total_stake += stake
            total_ev += ev * stake

        st.markdown(f"""
        | Top 5 Bets | Stake % | Expected Return |
        |------------|---------|-----------------|
        """)
        for pick in filtered_picks[:5]:
            stake_pct = pick['kelly_stake'] * 100
            ev_return = (pick['model_prob'] * pick['odds'] - 1) * stake_pct
            st.markdown(f"| {pick['horse']} @ {decimal_to_fractional(pick['odds'])} | {stake_pct:.1f}% | +{ev_return:.2f}% |")

    else:
        st.warning("No value bets found matching your criteria. Try lowering the thresholds.")

    # Expected value calculator
    st.subheader("🧮 Expected Value Calculator")

    col1, col2 = st.columns(2)

    with col1:
        calc_odds = st.number_input("Enter Odds (decimal format for calculation)", min_value=1.01, max_value=100.0, value=5.0, help="Enter in decimal format e.g. 4/1 = 5.0, 3/1 = 4.0, 2/1 = 3.0, Evens = 2.0")
        calc_prob = st.slider("Your Estimated Win Probability (%)", 1, 100, 25)
        calc_stake = st.number_input("Stake Amount (£)", min_value=1.0, max_value=10000.0, value=10.0)

    with col2:
        implied = 1 / calc_odds
        edge = (calc_prob / 100) - implied
        ev = (calc_prob / 100 * calc_odds * calc_stake) - calc_stake
        kelly = kelly_criterion(calc_prob / 100, calc_odds) * 100

        st.metric("Fractional Odds", decimal_to_fractional(calc_odds))
        st.metric("Implied Probability", f"{implied*100:.1f}%")
        st.metric("Your Edge", f"{edge*100:+.1f}%", delta="Value Bet!" if edge > 0 else "No Value")
        st.metric("Expected Value", f"£{ev:.2f}", delta="Profitable" if ev > 0 else "Loss Expected")
        st.metric("Recommended Stake (Kelly/4)", f"{kelly:.1f}% of bankroll")


elif page == "🎰 Accumulator Builder":
    st.markdown('<h1 class="main-header">🎰 Accumulator Builder</h1>', unsafe_allow_html=True)

    st.markdown("""
    Build accumulators with optimal expected value. The tool calculates combined probabilities
    and identifies the best accumulator combinations.
    """)

    st.markdown("---")

    # Accumulator type
    acca_type = st.radio(
        "Accumulator Type",
        ["Win Accumulator", "Place Accumulator (Each horse Top 3)", "Mixed (Win + Place)"],
        horizontal=True
    )

    st.markdown("---")

    # Selection builder
    st.subheader("📝 Build Your Accumulator")

    if 'acca_selections' not in st.session_state:
        st.session_state.acca_selections = []

    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

    with col1:
        new_horse = st.text_input("Horse Name", placeholder="Enter horse name")

    with col2:
        new_odds = st.number_input("Odds (decimal)", min_value=1.01, max_value=500.0, value=3.0, help="e.g. 2/1 = 3.0")

    with col3:
        new_prob = st.slider("Win Probability (%)", 5, 80, 35)

    with col4:
        st.write("")
        st.write("")
        if st.button("➕ Add", use_container_width=True):
            if new_horse:
                st.session_state.acca_selections.append({
                    'horse': new_horse,
                    'odds': new_odds,
                    'prob': new_prob / 100
                })

    # Display current selections
    if st.session_state.acca_selections:
        st.markdown("---")
        st.subheader("📋 Current Selections")

        for i, sel in enumerate(st.session_state.acca_selections):
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

            with col1:
                st.write(f"**{i+1}. {sel['horse']}**")
            with col2:
                st.write(f"Odds: {decimal_to_fractional(sel['odds'])}")
            with col3:
                st.write(f"Prob: {sel['prob']*100:.0f}%")
            with col4:
                if st.button("❌", key=f"remove_{i}"):
                    st.session_state.acca_selections.pop(i)
                    st.rerun()

        # Calculate accumulator stats
        st.markdown("---")
        st.subheader("📊 Accumulator Analysis")

        combined_odds = calculate_accumulator_odds([s['odds'] for s in st.session_state.acca_selections])
        combined_prob = calculate_accumulator_probability([s['prob'] for s in st.session_state.acca_selections])
        implied_prob = calculate_implied_probability(combined_odds)
        edge = combined_prob - implied_prob

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Combined Odds", decimal_to_fractional(combined_odds))

        with col2:
            st.metric("Win Probability", f"{combined_prob*100:.2f}%")

        with col3:
            st.metric("Implied Probability", f"{implied_prob*100:.2f}%")

        with col4:
            st.metric("Value Edge", f"{edge*100:+.2f}%", delta="✅ Value" if edge > 0 else "❌ No Value")

        # Returns calculator
        st.markdown("---")
        st.subheader("💰 Returns Calculator")

        stake = st.number_input("Enter Stake (£)", min_value=0.5, max_value=1000.0, value=10.0)

        col1, col2, col3 = st.columns(3)

        with col1:
            returns = stake * combined_odds
            st.metric("Potential Returns", f"£{returns:.2f}")

        with col2:
            profit = returns - stake
            st.metric("Potential Profit", f"£{profit:.2f}")

        with col3:
            ev = (combined_prob * returns) - stake
            st.metric("Expected Value", f"£{ev:.2f}", delta="Profitable" if ev > 0 else "Loss Expected")

        # Recommendation
        st.markdown("---")

        if edge > 0.05:
            st.success(f"""
            ✅ **RECOMMENDED BET**

            This accumulator has positive expected value with a {edge*100:.1f}% edge.

            Suggested stake: {kelly_criterion(combined_prob, combined_odds)*100:.1f}% of bankroll (Kelly/4)
            """)
        elif edge > 0:
            st.warning(f"""
            ⚠️ **MARGINAL VALUE**

            This accumulator has slight positive expected value ({edge*100:.2f}% edge).
            Consider reducing stake or adding/removing selections.
            """)
        else:
            st.error(f"""
            ❌ **NEGATIVE EXPECTED VALUE**

            This accumulator is expected to lose money long-term ({edge*100:.2f}% negative edge).
            Reconsider your selections or find better odds.
            """)

        if st.button("🗑️ Clear All Selections"):
            st.session_state.acca_selections = []
            st.rerun()

    else:
        st.info("Add selections above to build your accumulator")

    # Pre-built suggestions
    st.markdown("---")
    st.subheader("💡 Suggested Accumulators")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="value-bet">
            <h4>🎯 Conservative Treble</h4>
            <p>3 selections, high probability</p>
            <ul>
                <li>Combined Odds: ~8-15</li>
                <li>Win Probability: 8-15%</li>
                <li>Focus: Strong favorites with value</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="bet-card">
            <h4>💰 Value Four-Fold</h4>
            <p>4 selections, balanced risk/reward</p>
            <ul>
                <li>Combined Odds: ~20-50</li>
                <li>Win Probability: 3-8%</li>
                <li>Focus: Mix of favorites and value picks</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


elif page == "📈 ELO Rankings":
    st.markdown('<h1 class="main-header">📈 ELO Rankings</h1>', unsafe_allow_html=True)

    # Load real ELO data
    elo_data = load_elo_ratings()

    st.info(f"📊 ELO ratings calculated from **{len(elo_data.get('horse_elo', {})):,}** horses, "
            f"**{len(elo_data.get('jockey_elo', {})):,}** jockeys, "
            f"**{len(elo_data.get('trainer_elo', {})):,}** trainers across 1.66M race results")

    tab1, tab2, tab3 = st.tabs(["🏇 Horses", "👤 Jockeys", "🏆 Trainers"])

    with tab1:
        st.subheader("Top 50 Horses by ELO Rating")

        # Search functionality
        search_horse = st.text_input("🔍 Search for a horse", key="horse_search")

        horse_elo_dict = elo_data.get('horse_elo', {})
        if horse_elo_dict:
            sorted_horses = sorted(horse_elo_dict.items(), key=lambda x: x[1], reverse=True)

            if search_horse:
                # Filter by search
                filtered = [(h, e) for h, e in sorted_horses if search_horse.lower() in h.lower()]
                if filtered:
                    horses_elo = pd.DataFrame({
                        'Rank': range(1, len(filtered[:50]) + 1),
                        'Horse': [h for h, e in filtered[:50]],
                        'ELO': [int(e) for h, e in filtered[:50]]
                    })
                else:
                    st.warning(f"No horses found matching '{search_horse}'")
                    horses_elo = pd.DataFrame()
            else:
                horses_elo = pd.DataFrame({
                    'Rank': range(1, 51),
                    'Horse': [h for h, e in sorted_horses[:50]],
                    'ELO': [int(e) for h, e in sorted_horses[:50]]
                })

            if not horses_elo.empty:
                # Create ELO chart
                fig = px.bar(horses_elo.head(20), x='ELO', y='Horse', orientation='h',
                            color='ELO', color_continuous_scale='Greens',
                            title='Top 20 Horses by ELO Rating')
                fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(horses_elo, use_container_width=True, hide_index=True)
        else:
            st.warning("No horse ELO data available. Run the training script to generate ELO ratings.")

    with tab2:
        st.subheader("Top 50 Jockeys by ELO Rating")

        search_jockey = st.text_input("🔍 Search for a jockey", key="jockey_search")

        jockey_elo_dict = elo_data.get('jockey_elo', {})
        if jockey_elo_dict:
            sorted_jockeys = sorted(jockey_elo_dict.items(), key=lambda x: x[1], reverse=True)

            if search_jockey:
                filtered = [(j, e) for j, e in sorted_jockeys if search_jockey.lower() in j.lower()]
                if filtered:
                    jockeys_elo = pd.DataFrame({
                        'Rank': range(1, len(filtered[:50]) + 1),
                        'Jockey': [j for j, e in filtered[:50]],
                        'ELO': [int(e) for j, e in filtered[:50]]
                    })
                else:
                    st.warning(f"No jockeys found matching '{search_jockey}'")
                    jockeys_elo = pd.DataFrame()
            else:
                jockeys_elo = pd.DataFrame({
                    'Rank': range(1, 51),
                    'Jockey': [j for j, e in sorted_jockeys[:50]],
                    'ELO': [int(e) for j, e in sorted_jockeys[:50]]
                })

            if not jockeys_elo.empty:
                fig = px.bar(jockeys_elo.head(20), x='ELO', y='Jockey', orientation='h',
                            color='ELO', color_continuous_scale='Blues',
                            title='Top 20 Jockeys by ELO Rating')
                fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(jockeys_elo, use_container_width=True, hide_index=True)
        else:
            st.warning("No jockey ELO data available.")

    with tab3:
        st.subheader("Top 50 Trainers by ELO Rating")

        search_trainer = st.text_input("🔍 Search for a trainer", key="trainer_search")

        trainer_elo_dict = elo_data.get('trainer_elo', {})
        if trainer_elo_dict:
            sorted_trainers = sorted(trainer_elo_dict.items(), key=lambda x: x[1], reverse=True)

            if search_trainer:
                filtered = [(t, e) for t, e in sorted_trainers if search_trainer.lower() in t.lower()]
                if filtered:
                    trainers_elo = pd.DataFrame({
                        'Rank': range(1, len(filtered[:50]) + 1),
                        'Trainer': [t for t, e in filtered[:50]],
                        'ELO': [int(e) for t, e in filtered[:50]]
                    })
                else:
                    st.warning(f"No trainers found matching '{search_trainer}'")
                    trainers_elo = pd.DataFrame()
            else:
                trainers_elo = pd.DataFrame({
                    'Rank': range(1, 51),
                    'Trainer': [t for t, e in sorted_trainers[:50]],
                    'ELO': [int(e) for t, e in sorted_trainers[:50]]
                })

            if not trainers_elo.empty:
                fig = px.bar(trainers_elo.head(20), x='ELO', y='Trainer', orientation='h',
                            color='ELO', color_continuous_scale='Oranges',
                            title='Top 20 Trainers by ELO Rating')
                fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(trainers_elo, use_container_width=True, hide_index=True)
        else:
            st.warning("No trainer ELO data available.")


elif page == "📋 Bet Tracker":
    st.markdown('<h1 class="main-header">📋 Bet Tracker</h1>', unsafe_allow_html=True)

    st.markdown("Track your bets and monitor performance. Results update automatically.")

    # Load bet history from CSV
    bet_history_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'bet_history.csv')
    if os.path.exists(bet_history_path):
        history_df = pd.read_csv(bet_history_path)
    else:
        history_df = pd.DataFrame()

    if len(history_df) > 0:
        # Split into active/pending and settled
        pending_df = history_df[history_df['result'] == 'Pending']
        settled_df = history_df[history_df['result'] != 'Pending']

        # --- Overall Summary Metrics ---
        st.subheader("📊 Performance Summary")

        total_staked = float(history_df['stake'].sum())
        total_returns = float(history_df['returns'].sum())
        total_profit = float(history_df['profit'].sum())
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

        settled_count = len(settled_df)
        wins = len(settled_df[settled_df['result'] == 'Won']) if settled_count > 0 else 0
        places = len(settled_df[settled_df['result'] == 'Placed']) if settled_count > 0 else 0
        losses = len(settled_df[settled_df['result'] == 'Lost']) if settled_count > 0 else 0
        win_rate = (wins / settled_count * 100) if settled_count > 0 else 0
        itm_rate = ((wins + places) / settled_count * 100) if settled_count > 0 else 0

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Staked", f"£{total_staked:.2f}")
        with col2:
            st.metric("Total Returns", f"£{total_returns:.2f}")
        with col3:
            profit_color = "normal" if total_profit >= 0 else "inverse"
            st.metric("Profit/Loss", f"£{total_profit:+.2f}", delta=f"{roi:+.1f}% ROI")
        with col4:
            st.metric("Win Rate", f"{win_rate:.0f}%", delta=f"{wins}W / {losses}L")
        with col5:
            st.metric("Pending Bets", f"{len(pending_df)}")

        st.markdown("---")

        # --- Active / Pending Bets ---
        if len(pending_df) > 0:
            st.subheader("⏳ Active Bets")

            for _, bet in pending_df.iterrows():
                bet_type_badge = bet['bet_type']
                odds_display = bet['odds'] if bet['odds'] != 'SP' else 'SP'
                potential = f"£{float(bet['potential_returns']):.2f}" if float(bet['potential_returns']) > 0 else 'TBC'

                col1, col2, col3, col4, col5 = st.columns([3, 2, 1.5, 1.5, 1.5])
                with col1:
                    st.markdown(f"**🏇 {bet['horse']}**")
                with col2:
                    st.markdown(f"📍 {bet['course']} · {bet['time']}")
                with col3:
                    st.markdown(f"🎯 {bet_type_badge} @ **{odds_display}**")
                with col4:
                    st.markdown(f"💷 £{float(bet['stake']):.2f}")
                with col5:
                    st.markdown(f"🎰 Potential: **{potential}**")

            total_pending_stake = float(pending_df['stake'].sum())
            total_potential = float(pending_df['potential_returns'].sum())
            st.info(f"💰 **{len(pending_df)} active bets** · £{total_pending_stake:.2f} at risk · Up to £{total_potential:.2f} potential returns")

        st.markdown("---")

        # --- Settled Bets ---
        if len(settled_df) > 0:
            st.subheader("✅ Settled Bets")

            for _, bet in settled_df.iterrows():
                result = bet['result']
                profit = float(bet['profit'])
                position = bet.get('position', '')

                if result == 'Won':
                    icon = "✅"
                    result_text = f"**WON** (1st) · +£{profit:.2f}"
                elif result == 'Placed':
                    icon = "🥉"
                    pos_str = f"{int(position)}" if pd.notna(position) and str(position).replace('.','').isdigit() else ''
                    result_text = f"**PLACED** ({pos_str}) · £{profit:+.2f}"
                else:
                    icon = "❌"
                    pos_str = f"{int(position)}" if pd.notna(position) and str(position).replace('.','').isdigit() else ''
                    pos_display = f" (Finished {pos_str})" if pos_str else ""
                    result_text = f"**LOST**{pos_display} · -£{abs(profit):.2f}"

                col1, col2, col3, col4 = st.columns([3, 2.5, 2, 2])
                with col1:
                    st.markdown(f"{icon} **{bet['horse']}**")
                with col2:
                    st.markdown(f"📍 {bet['course']} · {bet['time']}")
                with col3:
                    st.markdown(f"🎯 {bet['bet_type']} @ {bet['odds']}")
                with col4:
                    st.markdown(result_text)

        st.markdown("---")

        # --- Cumulative Profit Chart ---
        settled_for_chart = history_df[history_df['result'] != 'Pending'].copy()
        if len(settled_for_chart) > 1:
            st.subheader("📈 Profit Over Time")
            settled_for_chart = settled_for_chart.reset_index(drop=True)
            settled_for_chart['cumulative_profit'] = settled_for_chart['profit'].astype(float).cumsum()
            settled_for_chart['bet_num'] = range(1, len(settled_for_chart) + 1)
            settled_for_chart['label'] = settled_for_chart['horse'].str.split(' \\(').str[0]

            fig = px.line(settled_for_chart, x='bet_num', y='cumulative_profit',
                         title='Cumulative Profit (Settled Bets)',
                         labels={'bet_num': 'Bet Number', 'cumulative_profit': 'Cumulative P&L (£)'},
                         hover_data=['label', 'profit'])
            fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
            fig.update_traces(line_color='#1a472a', line_width=3)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- Full Bet Table ---
        st.subheader("📋 All Bets")

        display_df = history_df[['date', 'horse', 'course', 'time', 'bet_type', 'odds', 'stake', 'potential_returns', 'result', 'position', 'returns', 'profit']].copy()
        display_df.columns = ['Date', 'Horse', 'Course', 'Time', 'Type', 'Odds', 'Stake', 'Pot. Returns', 'Result', 'Pos', 'Returns', 'Profit']

        # Color code results
        def style_result(val):
            if val == 'Won':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'Placed':
                return 'background-color: #fff3cd; color: #856404'
            elif val == 'Lost':
                return 'background-color: #f8d7da; color: #721c24'
            return ''

        st.dataframe(
            display_df.style.applymap(style_result, subset=['Result']),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No bets recorded yet. Bets will appear here once you start tracking.")


elif page == "⚙️ Settings":
    st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)

    st.subheader("💰 Bankroll Management")

    bankroll = st.number_input("Total Bankroll (£)", min_value=10.0, value=1000.0)
    max_stake_pct = st.slider("Maximum Stake (%)", 1, 20, 5)
    kelly_fraction = st.slider("Kelly Fraction", 0.1, 1.0, 0.25)

    st.info(f"""
    **Staking Rules:**
    - Maximum single bet: £{bankroll * max_stake_pct / 100:.2f} ({max_stake_pct}% of bankroll)
    - Kelly stakes will be multiplied by {kelly_fraction} for safety
    """)

    st.markdown("---")

    st.subheader("🔔 Alert Settings")

    value_threshold = st.slider("Minimum Value Edge for Alerts (%)", 0, 30, 10)
    min_odds = st.number_input("Minimum Odds", min_value=1.1, value=2.0)
    max_odds = st.number_input("Maximum Odds", min_value=2.0, value=20.0)

    st.markdown("---")

    st.subheader("📊 Model Information")

    st.markdown("""
    **Model:** Gradient Boosting Classifier

    **Training Data:**
    - 1,661,412 race results
    - 178,430 races
    - 187,937 unique horses
    - Date range: 2015-2025

    **Performance:**
    - Winner Accuracy: 35.2%
    - Top 3 Accuracy: 68.9%
    - AUC-ROC: 0.799

    **Key Features:**
    1. Composite ELO (Horse + Jockey + Trainer)
    2. Track/Distance/Going specific performance
    3. Recent form analysis
    4. Jockey-Trainer combination stats
    """)


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with 🏇 by Horse Racing Predictor")
st.sidebar.markdown("Model trained on 1.66M races")
