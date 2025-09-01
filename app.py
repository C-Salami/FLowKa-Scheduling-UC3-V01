# -*- coding: utf-8 -*-
"""
Scooter Wheels Scheduler — UC3 (Vis.js Timeline version)
- Beautiful, colorful interactive timeline using Vis.js via streamlit-timeline
- Click any bar to (de)select an ORDER; all its operations are highlighted
- Select 1 order -> delay/advance/move; Select 2 -> swap
- Order ID is displayed inside each bar (HTML content)
- Mic is a toggle (start/stop), voice commands applied on stop
- No forced reruns; no flicker/loop
"""

import os
import json
import re
from datetime import timedelta
from typing import Optional, Tuple

import pytz
from dateutil import parser as dtp

import streamlit as st
import pandas as pd
import numpy as np

# Vis.js timeline
from streamlit_timeline import timeline

# Voice (toggle)
from streamlit_mic_recorder import mic_recorder

# HTTP for Deepgram
import requests


# =============================================================================
# PAGE / KEYS / CONSTANTS
# =============================================================================
st.set_page_config(page_title="Scooter Wheels Scheduler — UC3 (Vis Timeline)", layout="wide")

for k in ("OPENAI_API_KEY", "DEEPGRAM_API_KEY"):
    try:
        os.environ[k] = os.environ.get(k) or st.secrets[k]  # type: ignore[attr-defined]
    except Exception:
        pass

DEFAULT_TZ = "Asia/Makassar"
TZ = pytz.timezone(DEFAULT_TZ)


# =============================================================================
# DATA LOADING (UC2-compatible)
# =============================================================================
def _oNNN_to_order(text: str) -> str:
    if isinstance(text, str):
        m = re.fullmatch(r"[Oo](\d{1,4})", text.strip())
        if m:
            return f"Order {int(m.group(1))}"
    return text

def _normalize_loaded_ids(df: pd.DataFrame, col: str = "order_id") -> pd.DataFrame:
    if col in df.columns:
        df[col] = df[col].apply(_oNNN_to_order)
    return df

@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    orders = pd.read_csv("data/scooter_orders.csv", parse_dates=["due_date"])
    sched  = pd.read_csv("data/scooter_schedule.csv", parse_dates=["start","end","due_date"])
    orders = _normalize_loaded_ids(orders, "order_id")
    sched  = _normalize_loaded_ids(sched,  "order_id")
    return orders, sched

orders, base_schedule = load_data()
if "schedule_df" not in st.session_state:
    st.session_state.schedule_df = base_schedule.copy()


# =============================================================================
# STATE (filters, selection, mic)
# =============================================================================
if "filters_open" not in st.session_state:
    st.session_state.filters_open = True
if "filt_max_orders" not in st.session_state:
    st.session_state.filt_max_orders = 12
if "filt_wheels" not in st.session_state:
    st.session_state.filt_wheels = sorted(base_schedule["wheel_type"].unique().tolist())
if "filt_machines" not in st.session_state:
    st.session_state.filt_machines = sorted(base_schedule["machine"].unique().tolist())
if "cmd_log" not in st.session_state:
    st.session_state.cmd_log = []

# mic & extraction
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "last_audio_fp" not in st.session_state:
    st.session_state.last_audio_fp = None
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = None
if "last_extraction" not in st.session_state:
    st.session_state.last_extraction = None

# selection
if "selected_orders" not in st.session_state:
    st.session_state.selected_orders = []   # max 2
if "last_clicked_item_id" not in st.session_state:
    st.session_state.last_clicked_item_id = None


# =============================================================================
# CSS / LAYOUT
# =============================================================================
sidebar_display = "block" if st.session_state.filters_open else "none"
st.markdown(f"""
<style>
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stSidebar"] {{ display: {sidebar_display}; }}
.block-container {{ padding-top: 6px; padding-bottom: 200px; }}

.topbar {{
  position: sticky; top: 0; z-index: 100;
  background: #fff; border-bottom: 1px solid #eee;
  padding: 8px 10px; margin-bottom: 6px;
}}
.topbar .inner {{ display: flex; justify-content: space-between; align-items: center; gap: 10px; }}
.topbar .title {{ font-weight: 600; font-size: 16px; }}
.topbar .btn {{
  background: #000; color: #fff; border: none; border-radius: 8px;
  padding: 6px 12px; font-weight: 600; cursor: pointer;
}}
.topbar .btn:hover {{ opacity: 0.9; }}

.bottom-wrap {{
  position: fixed; left: 0; right: 0; bottom: 0; z-index: 1000;
  background: linear-gradient(to top, rgba(255,255,255,0.98), rgba(255,255,255,0.65));
  padding: 18px 0 24px 0;
}}
.bottom-inner {{
  max-width: 1100px; margin: 0 auto; padding: 0 24px;
  display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 16px;
}}
.transcript-box {{
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu;
  background: #fafafa; border: 1px solid #eee; border-radius: 10px;
  padding: 8px 12px; min-height: 40px;
}}
.transcript-box.placeholder {{ color: #999; font-style: italic; }}

/* Vis.js item styling via classes */
.vis-item.vis-selected, .order-selected {{
  box-shadow: 0 0 0 2px rgba(0,0,0,0.35) inset !important;
  filter: saturate(1.2) brightness(1.05);
}}
.order-dim {{
  opacity: 0.35 !important;
}}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# TOP BAR + SIDEBAR
# =============================================================================
st.markdown('<div class="topbar"><div class="inner">', unsafe_allow_html=True)
st.markdown('<div class="title">Scooter Wheels Scheduler — UC3 (Vis Timeline)</div>', unsafe_allow_html=True)
toggle_label = "Hide Filters" if st.session_state.filters_open else "Show Filters"
if st.button(toggle_label, key="toggle_filters_btn"):
    st.session_state.filters_open = not st.session_state.filters_open
    st.rerun()
st.markdown('</div></div>', unsafe_allow_html=True)

if st.session_state.filters_open:
    with st.sidebar:
        st.header("Filters ⚙️")
        st.session_state.filt_max_orders = st.number_input(
            "Orders", 1, 100, value=int(st.session_state.filt_max_orders), step=1
        )
        wheels_all = sorted(base_schedule["wheel_type"].unique().tolist())
        st.session_state.filt_wheels = st.multiselect(
            "Wheel", wheels_all, default=st.session_state.filt_wheels or wheels_all
        )
        machines_all = sorted(base_schedule["machine"].unique().tolist())
        st.session_state.filt_machines = st.multiselect(
            "Machine", machines_all, default=st.session_state.filt_machines or machines_all
        )
        if st.button("Reset filters"):
            st.session_state.filt_max_orders = 12
            st.session_state.filt_wheels = wheels_all
            st.session_state.filt_machines = machines_all
            st.rerun()

        with st.expander(" Debug (voice & intent)"):
            st.caption("What Deepgram heard and what the extractor produced.")
            st.markdown("**Last Deepgram transcript:**")
            st.code(st.session_state.last_transcript or "—", language=None)

            st.markdown("**Last extraction (raw → payload):**")
            if st.session_state.last_extraction:
                st.write("Raw text sent to extractor:")
                st.code(st.session_state.last_extraction["raw"], language=None)
                st.write("Parsed payload:")
                st.json(st.session_state.last_extraction["payload"])
                st.write(f"Source: `{st.session_state.last_extraction.get('source','?')}`")
            else:
                st.write("—")

        with st.expander(" Command log (recent)"):
            if st.session_state.cmd_log:
                st.dataframe(pd.DataFrame(st.session_state.cmd_log[-10:]), use_container_width=True)
            else:
                st.caption("No commands yet.")


# =============================================================================
# FILTERED SLICE
# =============================================================================
sched_all = st.session_state.schedule_df.copy()
mask = sched_all["wheel_type"].isin(st.session_state.filt_wheels or sched_all["wheel_type"].unique()) & \
       sched_all["machine"].isin(st.session_state.filt_machines or sched_all["machine"].unique())
filtered = sched_all.loc[mask].copy()

order_priority = filtered.groupby("order_id", as_index=False)["start"].min().sort_values("start")
keep_ids = order_priority["order_id"].head(int(st.session_state.filt_max_orders)).tolist()
sched = filtered[filtered["order_id"].isin(keep_ids)].copy()

# =============================================================================
# COLOR MAP (stable per order)
# =============================================================================
def _palette():
    return [
        "#2E93FA","#66DA26","#546E7A","#E91E63","#FF9800","#00E396","#775DD0","#F9A3A4",
        "#F86624","#1B998B","#2E294E","#F46036","#5BC0EB","#9C27B0","#00B8D9","#36B37E",
        "#FF6F61","#4C78A8","#F58518","#54A24B","#EECA3B","#B279A2","#FF9DA6","#9CC8C4",
        "#E45756","#F2B701","#54A24B","#4C78A8","#72B7B2","#B279A2","#F58518","#E6E6E6",
    ]

def color_map(order_ids):
    pal = _palette()
    return {oid: pal[i % len(pal)] for i, oid in enumerate(order_ids)}

orders_sorted = sched.groupby("order_id")["start"].min().sort_values().index.tolist()
cmap = color_map(orders_sorted)


# =============================================================================
# SELECTION STRIP
# =============================================================================
with st.container():
    so = st.session_state.selected_orders
    cols = st.columns([1,4,1])
    with cols[1]:
        if so:
            st.markdown("🟩 **Selected**: " + "  •  ".join(f"`{o}`" for o in so))
        else:
            st.caption("Click an order to select (one for delay/move, two for swap).")
    with cols[2]:
        if so and st.button("Clear selection"):
            st.session_state.selected_orders = []
            st.session_state.last_clicked_item_id = None
            st.rerun()


# =============================================================================
# BUILD Vis.js GROUPS & ITEMS
# =============================================================================
# groups = machines
machines = sorted(sched["machine"].astype(str).unique().tolist())
groups = [{"id": m, "content": m} for m in machines]

# items = operations (with HTML content showing order id), each item must have id, group, start, end
# We'll also compute a CSS class for highlight/dim based on selection.
selected = set(st.session_state.selected_orders)

def item_class(oid: str) -> str:
    if not selected:
        return ""  # no dimming by default
    return "order-selected" if oid in selected else "order-dim"

items = []
for idx, row in sched.sort_values(["machine", "start", "end"]).iterrows():
    oid = str(row["order_id"])
    machine = str(row["machine"])
    start_iso = pd.to_datetime(row["start"]).isoformat()
    end_iso   = pd.to_datetime(row["end"]).isoformat()
    color = cmap.get(oid, "#888")

    # HTML content: colored chip with white order id
    content_html = f"""
    <div style="
        display:inline-block; padding:2px 6px; border-radius:6px;
        background:{color}; color:#fff; font-size:12px; font-weight:600;">
        {oid}
    </div>
    """

    items.append({
        "id": f"{oid}::{start_iso}",
        "group": machine,
        "start": start_iso,
        "end": end_iso,
        "content": content_html,
        "title": f"{oid} • {machine} • {row.get('operation','')}",
        "className": item_class(oid),
    })

# Vis options
options = {
    "stack": False,
    "horizontalScroll": True,
    "zoomKey": "ctrlKey",
    "maxHeight": "560px",
    "minHeight": "560px",
    "margin": {"item": 8, "axis": 12},
    "orientation": {"axis": "top"},
    "timeAxis": {"scale": "minute", "step": 30},
    "showMajorLabels": True,
    "showCurrentTime": False,
}

# Render timeline & capture selected item ids (returns a dict with 'id' on click)
if items:
    evt = timeline(items=items, groups=groups, options=options, height=600)
else:
    st.info("No operations match the current filters.")
    evt = None

# Update selection (no forced rerun)
if evt and isinstance(evt, dict) and "id" in evt and evt["id"]:
    item_id = evt["id"]                # "Order N::2025-08-31T10:00:00"
    oid = item_id.split("::", 1)[0]
    if item_id != st.session_state.last_clicked_item_id:
        st.session_state.last_clicked_item_id = item_id
        sel = st.session_state.selected_orders
        if oid in sel:
            sel = [x for x in sel if x != oid]      # toggle off
        else:
            sel = sel + [oid]
            if len(sel) > 2:
                sel = sel[-2:]
        st.session_state.selected_orders = sel
        # No st.rerun(); Streamlit will rerun naturally on interaction


# =============================================================================
# NLP + APPLY (UC2-compatible)
# =============================================================================
UNITS = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
         "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19}
TENS = {"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90}

def words_to_int(s: str) -> Optional[int]:
    s = s.lower().strip().replace("-", " ")
    if s in UNITS: return UNITS[s]
    if s in TENS: return TENS[s]
    parts = s.split()
    if len(parts) == 2 and parts[0] in TENS and parts[1] in UNITS:
        return TENS[parts[0]] + UNITS[parts[1]]
    if re.fullmatch(r"\d{1,4}", s):
        return int(s)
    return None

def normalize_order_name(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    legacy = re.search(r"\bO(\d{1,4})\b", text, flags=re.I)
    if legacy:
        return f"Order {int(legacy.group(1))}"
    m = re.search(r"\border\s*(?:#\s*)?([A-Za-z\-]+|\d{1,4})\b", text, flags=re.I)
    if not m:
        return None
    token = m.group(1)
    n = words_to_int(token)
    if n is None:
        digits = re.findall(r"\d+", token)
        if digits:
            n = int(digits[-1])
    return f"Order {n}" if n is not None else None

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["delay_order", "move_order", "swap_orders"]},
        "order_id": {"type": "string"},
        "order_id_2": {"type": "string"},
        "days": {"type": "number"},
        "hours": {"type": "number"},
        "minutes": {"type": "number"},
        "date": {"type": "string"},
        "time": {"type": "string"},
        "timezone": {"type": "string", "default": "Asia/Makassar"},
        "note": {"type": "string"}
    },
    "required": ["intent"],
    "additionalProperties": False
}

def _parse_duration_chunks(text: str):
    d = {"days":0.0,"hours":0.0,"minutes":0.0}
    def numtok(tok: str):
        tok = tok.strip().lower().replace(",", ".").replace("-", " ")
        try:
            return float(tok)
        except Exception:
            pass
        val = words_to_int(tok)
        return float(val) if val is not None else None
    for num, unit in re.findall(r"([\d\.,]+|\b[\w\-]+\b)\s*(days?|d|hours?|h|minutes?|mins?|m)\b", text, flags=re.I):
        n = numtok(num)
        if n is None: 
            continue
        u = unit.lower()
        if u.startswith("d"): d["days"] += n
        elif u.startswith("h"): d["hours"] += n
        else: d["minutes"] += n
    return d

def _extract_with_openai(user_text: str):
    from openai import OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    SYSTEM = (
        "Normalize factory scheduling edit commands. Return ONLY JSON per schema. "
        "Intents: delay_order, move_order, swap_orders. Order IDs like 'Order N'. "
        "Timezone Asia/Makassar. Default move time 08:00. Minutes allowed."
    )
    GUIDE = (
        '1) "delay order five one day" -> {"intent":"delay_order","order_id":"Order 5","days":1}\n'
        '2) "push order 9 by 1h 30m" -> {"intent":"delay_order","order_id":"Order 9","hours":1,"minutes":30}\n'
        '3) "move order twelve to Aug 30 09:13" -> {"intent":"move_order","order_id":"Order 12","date":"2025-08-30","time":"09:13"}\n'
        '4) "swap order 2 with order 3" -> {"intent":"swap_orders","order_id":"Order 2","order_id_2":"Order 3"}\n'
        '5) "advance order 8 by two days" -> {"intent":"delay_order","order_id":"Order 8","days":-2}\n'
        '6) "delay O071 by 2 hours" -> {"intent":"delay_order","order_id":"Order 71","hours":2}\n'
    )
    resp = client.responses.create(
        model="gpt-5.1",
        input=[{"role":"system","content":SYSTEM},
               {"role":"user","content":GUIDE},
               {"role":"user","content":user_text}],
        response_format={"type":"json_schema","json_schema":{"name":"Edit","schema":INTENT_SCHEMA}}
    )
    text = resp.output[0].content[0].text
    data = json.loads(text)
    for k in ("order_id", "order_id_2"):
        if k in data and isinstance(k, str):
            norm = normalize_order_name(data[k])
            if norm: data[k] = norm
    data["_source"] = "openai"
    return data

def _regex_fallback(user_text: str):
    t = user_text.strip()
    low = t.lower()

    # SWAP
    m = re.search(r"(?:^|\b)(swap|switch)\s+(order[^,;]*)", low)
    if m:
        ids = re.findall(r"\border\s*(?:#\s*)?([A-Za-z\-]+|\d{1,4})\b", low, flags=re.I)
        legacy = re.findall(r"\bO(\d{1,4})\b", low, flags=re.I)
        tokens = ids + legacy
        if len(tokens) >= 2:
            a = normalize_order_name(f"order {tokens[0]}")
            b = normalize_order_name(f"order {tokens[1]}")
            if a and b and a != b:
                return {"intent":"swap_orders","order_id":a,"order_id_2":b,"_source":"regex"}

    # DELAY / ADVANCE
    delay_sign = +1
    if re.search(r"\b(advance|bring\s+forward|pull\s+in)\b", low):
        delay_sign = -1
        low_norm = re.sub(r"\b(advance|bring\s+forward|pull\s+in)\b", "delay", low)
    else:
        low_norm = low

    target = normalize_order_name(low_norm) or normalize_order_name(low)
    m = re.search(r"\b(delay|push|postpone)\b.*?\bby\b\s+(.+)$", low_norm)
    if target and m:
        dur = _parse_duration_chunks(m.group(2))
        if any(v != 0 for v in dur.values()):
            return {
                "intent":"delay_order","order_id":target,
                "days":delay_sign*dur["days"],
                "hours":delay_sign*dur["hours"],
                "minutes":delay_sign*dur["minutes"],
                "_source":"regex"
            }

    m = re.search(r"\b(delay|push|postpone)\b.*?(days?|d|hours?|h|minutes?|mins?|m)\b", low_norm)
    if target and m:
        dur = _parse_duration_chunks(low_norm)
        if any(v != 0 for v in dur.values()):
            return {
                "intent":"delay_order","order_id":target,
                "days":delay_sign*dur["days"],
                "hours":delay_sign*dur["hours"],
                "minutes":delay_sign*dur["minutes"],
                "_source":"regex"
            }

    # MOVE
    m = re.search(r"\b(move|set|schedule)\b.*?\b(to|on)\s+(.+)", low)
    if m:
        target = target or normalize_order_name(low)
        when = m.group(3) if len(m.groups()) >= 3 else None
        if target and when:
            try:
                dt = dtp.parse(when, fuzzy=True)
                return {
                    "intent":"move_order","order_id":target,
                    "date":dt.date().isoformat(),"time":dt.strftime("%H:%M"),
                    "_source":"regex"
                }
            except Exception:
                pass

    # Fallback "one day"
    if target and re.search(r"\b(delay|push|postpone)\b.*\bone day\b", low):
        return {"intent":"delay_order","order_id":target,"days":1,"_source":"regex"}

    return {"intent":"unknown","raw":user_text,"_source":"regex"}

def extract_intent(user_text: str) -> dict:
    try:
        if os.getenv("OPENAI_API_KEY"):
            return _extract_with_openai(user_text)
    except Exception:
        pass
    return _regex_fallback(user_text)


# =============================================================================
# VALIDATE / APPLY (schedule editing)
# =============================================================================
def validate_intent(payload: dict, orders_df: pd.DataFrame, sched_df: pd.DataFrame):
    intent = payload.get("intent")

    def order_exists(oid: Optional[str]) -> bool:
        return bool(oid) and (orders_df["order_id"] == oid).any()

    if intent not in ("delay_order", "move_order", "swap_orders"):
        return False, "Unsupported intent"

    if intent in ("delay_order", "move_order", "swap_orders"):
        oid = payload.get("order_id")
        if not order_exists(oid):
            return False, f"Unknown order_id: {oid}"

    if intent == "swap_orders":
        oid2 = payload.get("order_id_2")
        if not order_exists(oid2):
            return False, f"Unknown order_id_2: {oid2}"
        if oid2 == payload.get("order_id"):
            return False, "Cannot swap the same order."
        return True, "ok"

    if intent == "delay_order":
        for k in ("days","hours","minutes"):
            if k in payload and payload[k] is not None:
                try:
                    payload[k] = float(payload[k])
                except Exception:
                    return False, f"{k.capitalize()} must be numeric."
        if not any(payload.get(k) for k in ("days","hours","minutes")):
            return False, "Delay needs a duration (days/hours/minutes)."
        return True, "ok"

    if intent == "move_order":
        date_str = payload.get("date")
        hhmm = payload.get("time") or "08:00"
        if not date_str:
            return False, "Move needs a date."
        try:
            dt = dtp.parse(f"{date_str} {hhmm}")
            if dt.tzinfo is None:
                dt = TZ.localize(dt)
            else:
                dt = dt.astimezone(TZ)
            payload["_target_dt"] = dt
        except Exception:
            return False, f"Unparseable datetime: {date_str} {hhmm}"
        return True, "ok"

    return False, "Invalid payload"

def _repack_touched_machines(s: pd.DataFrame, touched_orders):
    machines = s.loc[s["order_id"].isin(touched_orders), "machine"].unique().tolist()
    for m in machines:
        block_idx = s.index[s["machine"] == m]
        block = s.loc[block_idx].sort_values(["start", "end"]).copy()
        last_end = None
        for idx, row in block.iterrows():
            if last_end is not None and row["start"] < last_end:
                dur = row["end"] - row["start"]
                s.at[idx, "start"] = last_end
                s.at[idx, "end"] = last_end + dur
            last_end = s.at[idx, "end"]
    return s

def apply_delay(schedule_df: pd.DataFrame, order_id: str, *, days=0, hours=0, minutes=0) -> pd.DataFrame:
    s = schedule_df.copy()
    delta = timedelta(days=float(days or 0), hours=float(hours or 0), minutes=float(minutes or 0))
    mask = s["order_id"] == order_id
    s.loc[mask, "start"] = s.loc[mask, "start"] + delta
    s.loc[mask, "end"]   = s.loc[mask, "end"]   + delta
    return _repack_touched_machines(s, [order_id])

def apply_move(schedule_df: pd.DataFrame, order_id: str, target_dt) -> pd.DataFrame:
    s = schedule_df.copy()
    t0 = s.loc[s["order_id"] == order_id, "start"].min()
    delta = target_dt - t0
    days    = delta.days
    hours   = delta.seconds // 3600
    minutes = (delta.seconds % 3600) // 60
    return apply_delay(s, order_id, days=days, hours=hours, minutes=minutes)

def apply_swap(schedule_df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    s = schedule_df.copy()
    a0 = s.loc[s["order_id"] == a, "start"].min()
    b0 = s.loc[s["order_id"] == b, "start"].min()
    da, db = (b0 - a0), (a0 - b0)
    s = apply_delay(s, a, days=da.days, hours=da.seconds // 3600, minutes=(da.seconds % 3600)//60)
    s = apply_delay(s, b, days=db.days, hours=db.seconds // 3600, minutes=(db.seconds % 3600)//60)
    return s


# =============================================================================
# RESOLVE SELECTION FOR PARTIAL COMMANDS
# =============================================================================
def _resolve_selection_defaults(payload: dict, _transcript: Optional[str]) -> dict:
    sel = st.session_state.selected_orders or []
    latest = sel[-1] if sel else None
    two = sel[-2:] if len(sel) >= 2 else sel
    intent = payload.get("intent")

    if intent in ("delay_order", "move_order"):
        if not payload.get("order_id") and latest:
            payload["order_id"] = latest

    if intent == "swap_orders":
        if not payload.get("order_id") and not payload.get("order_id_2") and len(two) == 2:
            payload["order_id"], payload["order_id_2"] = two[0], two[1]
        elif not payload.get("order_id") and latest:
            payload["order_id"] = latest
        elif payload.get("order_id") and not payload.get("order_id_2") and len(two) == 2:
            other = two[0] if two[1] == payload["order_id"] else two[-1]
            if other != payload["order_id"]:
                payload["order_id_2"] = other
    return payload


# =============================================================================
# DEEPGRAM (toggle record) — START/STOP
# =============================================================================
def _deepgram_transcribe_bytes(wav_bytes: bytes, mimetype: str = "audio/wav") -> str:
    key = os.getenv("DEEPGRAM_API_KEY")
    if not key:
        raise RuntimeError("DEEPGRAM_API_KEY not set")
    url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&language=en"
    headers = {"Authorization": f"Token {key}", "Content-Type": mimetype}
    r = requests.post(url, headers=headers, data=wav_bytes, timeout=45)
    r.raise_for_status()
    j = r.json()
    return j["results"]["channels"][0]["alternatives"][0]["transcript"].strip()

def _process_and_apply(cmd_text: str, *, source_hint: Optional[str] = None):
    from copy import deepcopy
    try:
        payload = extract_intent(cmd_text)
        payload = _resolve_selection_defaults(payload, cmd_text)

        st.session_state.last_extraction = {
            "raw": cmd_text,
            "payload": deepcopy(payload),
            "source": source_hint or payload.get("_source", "?"),
        }

        ok, msg = validate_intent(payload, orders, st.session_state.schedule_df)

        log_payload = deepcopy(payload)
        if "_target_dt" in log_payload:
            log_payload["_target_dt"] = str(log_payload["_target_dt"])
        st.session_state.cmd_log.append({
            "raw": cmd_text,
            "payload": log_payload,
            "ok": bool(ok),
            "msg": msg,
            "source": source_hint or payload.get("_source", "?")
        })
        st.session_state.cmd_log = st.session_state.cmd_log[-50:]

        if not ok:
            st.error(f"❌ Cannot apply: {msg}")
            return

        if payload["intent"] == "delay_order":
            st.session_state.schedule_df = apply_delay(
                st.session_state.schedule_df,
                payload["order_id"],
                days=payload.get("days", 0),
                hours=payload.get("hours", 0),
                minutes=payload.get("minutes", 0),
            )
            st.success(f"✅ Delayed {payload['order_id']}.")

        elif payload["intent"] == "move_order":
            st.session_state.schedule_df = apply_move(
                st.session_state.schedule_df,
                payload["order_id"],
                payload["_target_dt"],
            )
            st.success(f"✅ Moved {payload['order_id']}.")

        elif payload["intent"] == "swap_orders":
            st.session_state.schedule_df = apply_swap(
                st.session_state.schedule_df,
                payload["order_id"],
                payload["order_id_2"],
            )
            st.success(f"✅ Swapped {payload['order_id']} and {payload['order_id_2']}.")

        st.rerun()
    except Exception as e:
        st.error(f"⚠️ Error: {e}")


# =============================================================================
# BOTTOM MIC AREA (toggle)
# =============================================================================
with st.container():
    st.markdown('<div class="bottom-wrap"><div class="bottom-inner">', unsafe_allow_html=True)

    btn_label = "🛑 Stop" if st.session_state.is_recording else "🎙️ Start"
    if st.button(btn_label, key="toggle_mic_btn"):
        st.session_state.is_recording = not st.session_state.is_recording
        st.rerun()

    rec = None
    if st.session_state.is_recording:
        rec = mic_recorder(
            start_prompt="", stop_prompt="",
            key="mic_toggle",
            just_once=False, format="wav",
        )

    if st.session_state.last_transcript:
        st.markdown(f'<div class="transcript-box">{st.session_state.last_transcript}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="transcript-box placeholder">(your transcript will appear here)</div>', unsafe_allow_html=True)

    st.markdown(
        '<div style="white-space:nowrap; font-size:13px; opacity:0.75;">'
        'Click orders to select (1 for delay/move; 2 for swap). Speak while recording, then click Stop.'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div></div>', unsafe_allow_html=True)

# On Stop -> bytes arrive once
if rec and isinstance(rec, dict) and rec.get("bytes"):
    wav_bytes = rec["bytes"]
    fp = (len(wav_bytes), hash(wav_bytes[:1024]))
    if fp != st.session_state.last_audio_fp:
        st.session_state.last_audio_fp = fp
        try:
            with st.spinner("Transcribing…"):
                transcript = _deepgram_transcribe_bytes(wav_bytes, mimetype="audio/wav")
            st.session_state.last_transcript = transcript
            if transcript:
                _process_and_apply(transcript, source_hint="voice/deepgram")
            else:
                st.warning("No speech detected.")
        except Exception as e:
            st.error(f"Transcription failed: {e}")


# =============================================================================
# Manual text command (debug)
# =============================================================================
with st.expander("Manual command (debug)"):
    t = st.text_input("Type a command (e.g., 'delay this order by 2 days', 'advance this order 1 hour', 'swap orders')")
    if st.button("Apply command", type="primary"):
        if t.strip():
            _process_and_apply(t.strip(), source_hint="text/debug")
        else:
            st.info("Type something first.")
