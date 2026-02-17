"""
frontend/dashboard.py
─────────────────────
Streamlit "Command & Control" dashboard for the 360° Surveillance System.

Run with:
    streamlit run frontend/dashboard.py --server.port 8501

Features
--------
  • Dark-mode theme (via .streamlit/config.toml)
  • 2x2 camera grid (N/S/E/W views) with MJPEG feeds
  • st.metric cards for live counts
  • Red-Alert intrusion popup (st.error banner + st.toast)
  • Triage tabs: Human & PPE | Vehicle & Plates | Pet & Animal
  • Acknowledge button to dismiss the intrusion banner
  • Configuration sidebar with confidence slider
  • Auto-refresh loop (2 s)
"""

from __future__ import annotations

import time
import requests
import streamlit as st
import pandas as pd
from config.settings import CAMERAS, FASTAPI_PORT, LABEL_ICONS

# ════════════════════════════════════════════════════════════════════
#  Constants
# ════════════════════════════════════════════════════════════════════

API_BASE = f"http://localhost:{FASTAPI_PORT}"
REFRESH_INTERVAL = 2


# ════════════════════════════════════════════════════════════════════
#  Page config (must be first Streamlit call)
# ════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Command & Control — 360° Surveillance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ════════════════════════════════════════════════════════════════════
#  Session state defaults
# ════════════════════════════════════════════════════════════════════

if "alert_acknowledged" not in st.session_state:
    st.session_state.alert_acknowledged = False
if "event_cache" not in st.session_state:
    st.session_state.event_cache = []


# ════════════════════════════════════════════════════════════════════
#  Helper: safe API calls
# ════════════════════════════════════════════════════════════════════

def api_get(path: str, default=None, timeout: float = 3.0):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return default


def api_post(path: str, json_body: dict, timeout: float = 3.0):
    try:
        r = requests.post(f"{API_BASE}{path}", json=json_body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════
#  RED-ALERT Intrusion Banner
# ════════════════════════════════════════════════════════════════════

def render_intrusion_alert():
    """
    Checks /metadata for the intrusion_active flag.
    If active and not acknowledged:
      - Shows a persistent st.error banner at the top
      - Fires st.toast for immediate visual feedback
    """
    meta = api_get("/metadata", {})
    intrusion_active = meta.get("intrusion_active", False) if meta else False
    latest = meta.get("latest_alert") if meta else None

    if intrusion_active and not st.session_state.alert_acknowledged:
        # Persistent Red-Alert banner
        alert_col, btn_col = st.columns([5, 1])
        with alert_col:
            detail = ""
            if latest:
                detail = f"  |  {latest.get('camera', '?')} — {latest.get('detail', '')}"
            st.error(
                f"🚨 **RED ALERT — INTRUSION DETECTED**{detail}  \n"
                "A target has entered a restricted zone. Review camera feeds immediately.",
                icon="🚨",
            )
        with btn_col:
            st.markdown("")  # spacer
            if st.button("✅ Acknowledge", type="primary", use_container_width=True):
                st.session_state.alert_acknowledged = True
                st.rerun()

        # Toast notification
        if latest:
            st.toast(
                f"🚨 Intrusion: {latest.get('camera', '?')} — {latest.get('detail', '')}",
                icon="🚨",
            )
    elif not intrusion_active:
        # Reset ack when the intrusion clears
        st.session_state.alert_acknowledged = False


# ════════════════════════════════════════════════════════════════════
#  Sidebar — Configuration
# ════════════════════════════════════════════════════════════════════

def render_sidebar():
    st.sidebar.title("⚙️ Configuration")

    # Confidence slider
    current_conf = api_get("/config/confidence", {"confidence": 0.45})
    conf_val = current_conf.get("confidence", 0.45) if current_conf else 0.45

    new_conf = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.05,
        max_value=1.0,
        value=float(conf_val),
        step=0.05,
        help="Lower = more detections (more false-positives). Higher = fewer but more certain.",
    )
    if abs(new_conf - conf_val) > 0.01:
        api_post("/config/confidence", {"value": new_conf})
        st.sidebar.success(f"Confidence updated to {new_conf:.2f}")

    auto_refresh = st.sidebar.checkbox("Auto-refresh (2 s)", value=True)

    st.sidebar.markdown("---")

    # Sidebar alerts
    st.sidebar.title("🚨 Recent Alerts")
    alerts = api_get("/alerts?n=20", [])
    if alerts:
        for a in alerts[:10]:
            icon = LABEL_ICONS.get(a.get("label", ""), "ℹ️")
            st.sidebar.markdown(
                f"**{icon} {a['time']}** — *{a['camera']}*  \n"
                f"{a['detail']}"
            )
    else:
        st.sidebar.info("No alerts yet.")

    # System status
    st.sidebar.markdown("---")
    health = api_get("/", {})
    if health and health.get("status") == "running":
        st.sidebar.success(f"Backend online — {health.get('cameras', 0)} cameras")
    else:
        st.sidebar.error("Backend offline")

    return auto_refresh


# ════════════════════════════════════════════════════════════════════
#  Summary Metrics Row
# ════════════════════════════════════════════════════════════════════

def render_summary_metrics():
    stats = api_get("/stats", {})
    if not stats:
        st.warning("Waiting for backend to produce stats…")
        return

    total_people    = sum(s.get("detections", {}).get("Person", 0) for s in stats.values())
    total_vehicles  = sum(s.get("detections", {}).get("Vehicle", 0) for s in stats.values())
    total_pets      = sum(s.get("detections", {}).get("Pet", 0) for s in stats.values())
    total_intrusions = sum(s.get("intrusions", 0) for s in stats.values())
    avg_fps         = sum(s.get("fps", 0) for s in stats.values()) / max(len(stats), 1)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🧑 Active People", total_people)
    m2.metric("🚗 Vehicles", total_vehicles)
    m3.metric("🐾 Pets", total_pets)
    m4.metric("🚨 Total Intrusions", total_intrusions)
    m5.metric("⚡ Avg FPS", f"{avg_fps:.1f}")


# ════════════════════════════════════════════════════════════════════
#  2x2 Camera Grid (N/S/E/W)
# ════════════════════════════════════════════════════════════════════

def render_camera_grid():
    st.subheader("📹 360° Camera Grid")

    stats = api_get("/stats", {})

    # Force 2x2 layout
    rows = [CAMERAS[i:i + 2] for i in range(0, len(CAMERAS), 2)]

    for row_cameras in rows:
        cols = st.columns(len(row_cameras))
        for col, cam in zip(cols, row_cameras):
            with col:
                cam_stat = stats.get(cam.cam_id, {})
                fps    = cam_stat.get("fps", 0)
                det    = cam_stat.get("detections", {})
                intr   = cam_stat.get("intrusions", 0)

                # Header row
                status = "🟢" if fps > 0 else "🔴"
                st.markdown(f"#### {status} {cam.label or cam.cam_id}")

                # MJPEG stream
                stream_url = f"{API_BASE}/video/{cam.cam_id}?fps=12"
                st.markdown(
                    f'<img src="{stream_url}" width="100%" '
                    f'style="border-radius:8px; border:1px solid #333;" />',
                    unsafe_allow_html=True,
                )

                # Stats line
                det_parts = []
                for lbl, cnt in det.items():
                    icon = LABEL_ICONS.get(lbl, "")
                    det_parts.append(f"{icon} {lbl}: **{cnt}**")
                det_str = "  ·  ".join(det_parts) if det_parts else "No detections"

                intr_badge = f"🚨 **{intr}** intrusions" if intr else "✅ Clear"
                st.markdown(f"**{fps} FPS**  |  {intr_badge}")
                st.caption(det_str)


# ════════════════════════════════════════════════════════════════════
#  Triage Tabs — Event Segregation
# ════════════════════════════════════════════════════════════════════

def render_triage_tabs():
    st.subheader("📊 Event Triage")

    tab_human, tab_vehicle, tab_pet = st.tabs([
        "🧑 Human Traffic & PPE",
        "🚗 Vehicle & Plates",
        "🐾 Pets & Animals",
    ])

    # Fetch all events once and store in session_state for persistence
    all_events = api_get("/events?n=200", [])
    if all_events:
        st.session_state.event_cache = all_events
    events = st.session_state.event_cache

    with tab_human:
        human_events = [e for e in events if e.get("category") == "Human & PPE"]
        if human_events:
            df = pd.DataFrame(human_events)
            cols_to_show = [c for c in ["time", "camera", "label", "confidence", "intrusion", "detail"] if c in df.columns]
            st.dataframe(df[cols_to_show], width='stretch', height=300)
            st.caption(f"{len(human_events)} events")
        else:
            st.info("No human / PPE events recorded yet.")

    with tab_vehicle:
        vehicle_events = [e for e in events if e.get("category") == "Vehicle & Plates"]
        if vehicle_events:
            df = pd.DataFrame(vehicle_events)
            cols_to_show = [c for c in ["time", "camera", "label", "confidence", "detail"] if c in df.columns]
            st.dataframe(df[cols_to_show], width='stretch', height=300)
            st.caption(f"{len(vehicle_events)} events")
        else:
            st.info("No vehicle / plate events recorded yet.")

    with tab_pet:
        pet_events = [e for e in events if e.get("category") == "Pet & Animal"]
        if pet_events:
            df = pd.DataFrame(pet_events)
            cols_to_show = [c for c in ["time", "camera", "label", "confidence", "detail"] if c in df.columns]
            st.dataframe(df[cols_to_show], width='stretch', height=300)
            st.caption(f"{len(pet_events)} events")
        else:
            st.info("No pet / animal events recorded yet.")


# ════════════════════════════════════════════════════════════════════
#  Full Event Log
# ════════════════════════════════════════════════════════════════════

def render_event_log():
    with st.expander("📋 Full Event Log (all categories)", expanded=False):
        events = st.session_state.event_cache
        if events:
            df = pd.DataFrame(events)
            st.dataframe(df, width='stretch', height=350)
        else:
            st.info("No events recorded yet.")


# ════════════════════════════════════════════════════════════════════
#  Entrypoint
# ════════════════════════════════════════════════════════════════════

def main():
    auto_refresh = render_sidebar()

    st.title("🛡️ Command & Control — 360° Surveillance")
    st.caption(
        "YOLOv8n + OpenVINO  ·  Real-time intrusion detection  ·  Event triage dashboard"
    )

    # Red-Alert banner (top of page, impossible to miss)
    render_intrusion_alert()

    # Metric cards
    render_summary_metrics()

    st.markdown("---")

    # 2x2 camera grid
    render_camera_grid()

    st.markdown("---")

    # Triage tabs
    render_triage_tabs()

    # Full log (collapsed by default)
    render_event_log()

    # Auto-refresh
    if auto_refresh:
        time.sleep(REFRESH_INTERVAL)
        st.rerun()


if __name__ == "__main__":
    main()
else:
    main()
