import streamlit as st
from cassandra.cluster import Cluster
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
import time

# ======================================================
# PAGE CONFIGURATION
# ======================================================
st.set_page_config(
    page_title="Healthcare Monitoring System",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .status-critical {
        background-color: #fee2e2;
        border-left-color: #ef4444;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid;
    }
    .status-normal {
        background-color: #dcfce7;
        border-left-color: #22c55e;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid;
    }
    .section-divider {
        margin: 2rem 0;
        border-top: 2px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# DATABASE CONNECTION
# ======================================================
@st.cache_resource
def get_session():
    try:
        cluster = Cluster(["127.0.0.1"])
        session = cluster.connect("healthcare")
        return session
    except Exception as e:
        st.error(f"Database Connection Failed: {e}")
        return None

session = get_session()

# ======================================================
# DATA RETRIEVAL WITH LIMIT (OPTIMIZED)
# ======================================================
@st.cache_data(ttl=2)  # Cache selama 2 detik untuk mengurangi query ke DB
def fetch_data(p_id, minutes=30):
    if not session: 
        return pd.DataFrame(), pd.DataFrame()
    
    since = datetime.utcnow() - timedelta(minutes=minutes)
    since_str = since.strftime('%Y-%m-%d %H:%M:%S')
    
    # TAMBAHKAN LIMIT untuk membatasi data yang diambil
    query = f"""
        SELECT * FROM vital_predictions 
        WHERE patient_id = {p_id} 
        AND event_time >= '{since_str}'
        LIMIT 500
    """
    rows = session.execute(query)
    
    # Pilih hanya kolom yang diperlukan (jangan ambil semua)
    needed_cols = [
        'patient_id', 'event_time', 'heart_rate', 'respiratory_rate', 
        'body_temperature', 'oxygen_saturation', 'systolic_bp', 'diastolic_bp',
        'age', 'gender', 'weight_kg', 'height_m', 'derived_hrv', 
        'derived_pulse_pressure', 'derived_bmi', 'derived_map',
        'prediction', 'probability_high_risk'
    ]
    
    df = pd.DataFrame(rows)
    
    if df.empty: 
        return pd.DataFrame(), pd.DataFrame()
    
    # Filter hanya kolom yang ada
    available_cols = [col for col in needed_cols if col in df.columns]
    df = df[available_cols]
    
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values("event_time", ascending=True)
    
    # DOWNSAMPLING: Ambil maksimal 100 data point untuk chart (hindari overplotting)
    if len(df) > 100:
        # Ambil data terakhir sebagai latest
        latest_df = df.tail(1)
        # Sample data untuk history chart agar tidak terlalu padat
        step = len(df) // 100
        history_df = df.iloc[::step].copy()
        # Pastikan data terakhir tetap ada
        if history_df.iloc[-1]['event_time'] != df.iloc[-1]['event_time']:
            history_df = pd.concat([history_df, df.tail(1)])
    else:
        latest_df = df.tail(1)
        history_df = df
    
    return latest_df, history_df

# ======================================================
# SIDEBAR CONFIGURATION
# ======================================================
with st.sidebar:
    st.markdown("### System Configuration")
    st.markdown("---")
    
    patient_id = st.selectbox(
        "Patient ID",
        options=range(1, 6),
        help="Select patient to monitor"
    )
    
    history_range = st.slider(
        "Historical Data Range (Minutes)",
        min_value=5,
        max_value=60,
        value=30,
        help="Time range for trend visualization"
    )
    
    refresh_rate = st.slider(
        "Auto-Refresh Interval (Seconds)",
        min_value=2,
        max_value=10,
        value=3,  # Naikkan default dari 2 ke 3 detik
        help="Dashboard refresh frequency"
    )
    
    st.markdown("---")
    st.markdown("### System Status")
    st.success("Connected to Database")
    st.info(f"Monitoring Patient {patient_id}")

# Auto-refresh mechanism
st_autorefresh(interval=refresh_rate * 1000, key="global_refresh")

# ======================================================
# MAIN DASHBOARD
# ======================================================
st.markdown('<p class="main-header">Healthcare Monitoring System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-Time Patient Vital Signs and Risk Assessment</p>', unsafe_allow_html=True)

latest_df, history_df = fetch_data(patient_id, minutes=history_range)

if latest_df.empty:
    st.warning(f"Waiting for data stream from Patient {patient_id}. Please ensure the streaming service is active.")
    st.info("Check that Apache Spark Streaming is running and publishing data to the database.")
else:
    curr = latest_df.iloc[0]

    # ======================================================
    # RISK STATUS INDICATOR
    # ======================================================
    prob_pct = curr['probability_high_risk'] * 100
    
    if curr['prediction'] == 1:
        st.markdown(
            f'<div class="status-critical"><strong>ALERT: HIGH RISK DETECTED</strong><br/>Risk Probability: {prob_pct:.1f}% | Immediate attention recommended</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="status-normal"><strong>STATUS: NORMAL</strong><br/>Risk Probability: {prob_pct:.1f}% | Patient condition stable</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ======================================================
    # PRIMARY VITAL SIGNS METRICS
    # ======================================================
    st.markdown("### Current Vital Signs")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Calculate delta for heart rate
    delta_hr = 0
    if len(history_df) > 1:
        delta_hr = curr['heart_rate'] - history_df.iloc[-2]['heart_rate']

    with col1:
        st.metric(
            label="Heart Rate",
            value=f"{curr['heart_rate']:.0f}",
            delta=f"{delta_hr:.1f} BPM" if delta_hr != 0 else None,
            help="Beats per minute"
        )
    
    with col2:
        st.metric(
            label="Oxygen Saturation",
            value=f"{curr['oxygen_saturation']:.1f}%",
            help="SpO2 level"
        )
    
    with col3:
        st.metric(
            label="Blood Pressure",
            value=f"{curr['systolic_bp']:.0f}/{curr['diastolic_bp']:.0f}",
            help="Systolic/Diastolic mmHg"
        )
    
    with col4:
        st.metric(
            label="Body Temperature",
            value=f"{curr['body_temperature']:.1f}°C",
            help="Core temperature"
        )
    
    with col5:
        st.metric(
            label="Body Mass Index",
            value=f"{curr['derived_bmi']:.1f}",
            help="BMI calculation"
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ======================================================
    # PATIENT DEMOGRAPHICS
    # ======================================================
    st.markdown("### Patient Demographics")
    
    demo_col1, demo_col2, demo_col3, demo_col4, demo_col5 = st.columns(5)
    
    with demo_col1:
        st.markdown(f"**Gender:** {curr['gender']}")
    
    with demo_col2:
        st.markdown(f"**Age:** {curr['age']} years")
    
    with demo_col3:
        weight = f"{curr['weight_kg']:.1f} kg" if pd.notnull(curr['weight_kg']) else "Not Available"
        st.markdown(f"**Weight:** {weight}")
    
    with demo_col4:
        height = f"{curr['height_m']:.2f} m" if pd.notnull(curr['height_m']) else "Not Available"
        st.markdown(f"**Height:** {height}")
    
    with demo_col5:
        st.markdown(f"**Respiratory Rate:** {curr['respiratory_rate']:.0f} bpm")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ======================================================
    # DERIVED CLINICAL INDICATORS
    # ======================================================
    st.markdown("### Derived Clinical Indicators")
    
    derived_col1, derived_col2, derived_col3, derived_col4 = st.columns(4)
    
    with derived_col1:
        st.metric("Heart Rate Variability", f"{curr['derived_hrv']:.2f}", help="HRV metric")
    
    with derived_col2:
        st.metric("Pulse Pressure", f"{curr['derived_pulse_pressure']:.0f} mmHg", help="Systolic - Diastolic")
    
    with derived_col3:
        st.metric("Mean Arterial Pressure", f"{curr['derived_map']:.0f} mmHg", help="MAP calculation")
    
    with derived_col4:
        st.metric("Risk Score", f"{prob_pct:.1f}%", help="AI-predicted risk probability")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ======================================================
    # HISTORICAL TREND ANALYSIS (OPTIMIZED CHARTS)
    # ======================================================
    st.markdown("### Trend Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Cardiovascular Metrics",
        "Blood Pressure Analysis",
        "Risk Assessment",
        "Respiratory & Temperature"
    ])
    
    # Gunakan config untuk disable beberapa fitur plotly yang tidak perlu
    chart_config = {
        'displayModeBar': False,  # Sembunyikan toolbar
        'staticPlot': False
    }

    with tab1:
        fig_cardio = go.Figure()
        # Gunakan mode 'lines' saja tanpa markers untuk performa lebih baik
        fig_cardio.add_trace(go.Scatter(
            x=history_df["event_time"],
            y=history_df["heart_rate"],
            mode='lines',
            name='Heart Rate (BPM)',
            line=dict(color='#ef4444', width=2)
        ))
        fig_cardio.add_trace(go.Scatter(
            x=history_df["event_time"],
            y=history_df["oxygen_saturation"],
            mode='lines',
            name='SpO2 (%)',
            line=dict(color='#3b82f6', width=2),
            yaxis='y2'
        ))
        fig_cardio.update_layout(
            title="Cardiovascular Metrics Over Time",
            xaxis_title="Time",
            yaxis_title="Heart Rate (BPM)",
            yaxis2=dict(title="SpO2 (%)", overlaying='y', side='right'),
            template="plotly_white",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_cardio, use_container_width=True, config=chart_config)

    with tab2:
        fig_bp = go.Figure()
        fig_bp.add_trace(go.Scatter(
            x=history_df["event_time"],
            y=history_df["systolic_bp"],
            mode='lines',
            name='Systolic',
            line=dict(color='#dc2626', width=2)
        ))
        fig_bp.add_trace(go.Scatter(
            x=history_df["event_time"],
            y=history_df["diastolic_bp"],
            mode='lines',
            name='Diastolic',
            line=dict(color='#2563eb', width=2)
        ))
        fig_bp.add_trace(go.Scatter(
            x=history_df["event_time"],
            y=history_df["derived_map"],
            mode='lines',
            name='MAP',
            line=dict(color='#059669', width=2, dash='dash')
        ))
        fig_bp.update_layout(
            title="Blood Pressure Trends",
            xaxis_title="Time",
            yaxis_title="Pressure (mmHg)",
            template="plotly_white",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_bp, use_container_width=True, config=chart_config)

    with tab3:
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Scatter(
            x=history_df["event_time"],
            y=history_df["probability_high_risk"] * 100,
            mode='lines',
            name='Risk Probability',
            fill='tozeroy',
            line=dict(color='#dc2626', width=2)
        ))
        fig_risk.add_hline(y=50, line_dash="dash", line_color="orange", 
                          annotation_text="Threshold: 50%")
        fig_risk.update_layout(
            title="AI Risk Assessment Timeline",
            xaxis_title="Time",
            yaxis_title="Risk Probability (%)",
            template="plotly_white",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_risk, use_container_width=True, config=chart_config)

    with tab4:
        fig_resp = go.Figure()
        fig_resp.add_trace(go.Scatter(
            x=history_df["event_time"],
            y=history_df["respiratory_rate"],
            mode='lines',
            name='Respiratory Rate',
            line=dict(color='#8b5cf6', width=2)
        ))
        fig_resp.add_trace(go.Scatter(
            x=history_df["event_time"],
            y=history_df["body_temperature"],
            mode='lines',
            name='Temperature',
            line=dict(color='#f59e0b', width=2),
            yaxis='y2'
        ))
        fig_resp.update_layout(
            title="Respiratory Rate and Body Temperature",
            xaxis_title="Time",
            yaxis_title="Respiratory Rate (bpm)",
            yaxis2=dict(title="Temperature (°C)", overlaying='y', side='right'),
            template="plotly_white",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_resp, use_container_width=True, config=chart_config)

    # ======================================================
    # RAW DATA LOG (OPTIMIZED - TAMPILKAN 50 DATA TERAKHIR SAJA)
    # ======================================================
    with st.expander("View Detailed Data Log"):
        display_cols = [
            'event_time', 'heart_rate', 'respiratory_rate', 'body_temperature',
            'oxygen_saturation', 'systolic_bp', 'diastolic_bp', 
            'prediction', 'probability_high_risk'
        ]
        available_cols = [col for col in display_cols if col in history_df.columns]
        # Tampilkan hanya 50 data terakhir
        st.dataframe(
            history_df[available_cols].sort_values("event_time", ascending=False).head(50),
            use_container_width=True,
            hide_index=True
        )
        st.caption(f"Showing last 50 records out of {len(history_df)} total records")

# ======================================================
# FOOTER
# ======================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption(f"System Time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

with footer_col2:
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with footer_col3:
    st.caption(f"Data Points: {len(history_df)} | Refresh: {refresh_rate}s")