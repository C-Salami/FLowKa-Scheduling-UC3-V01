import os
import json
import re
from datetime import timedelta
import pytz
from dateutil import parser as dtp

import streamlit as st
import pandas as pd
import altair as alt

# Single-icon, press-and-hold microphone component
from streamlit_mic_recorder import mic_recorder

# ============================ PAGE & SECRETS ============================
st.set_page_config(page_title="Scooter Wheels Scheduler", layout="wide")

# Pull keys from Streamlit secrets if present
for k in ("OPENAI_API_KEY", "DEEPGRAM_API_KEY"):
    try:
        os.environ[k] = os.environ.get(k) or st.secrets[k]
    except Exception:
        pass  # ok

# ============================ DATA LOADING =============================
@st.cache_data
def load_data():
    # expects CSVs created by generate_sample_data.py inside ./data
    orders = pd.read_csv("data/scooter_orders.csv", parse_dates=["due_date"])
    sched = pd.read_csv("data/scooter_schedule.csv", parse_dates=["start", "end", "due_date"])
    return orders, sched

orders, base_schedule = load_data()
if "schedule_df" not in st.session_state:
    st.session_state.schedule_df = base_schedule.copy()

# ============================ STATE ============================
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

# ----- prompt & voice pipeline state -----
if "prompt_text" not in st.session_state:
    st.session_state.prompt_text = ""           # current visible text in the prompt bar
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None      # text to inject (e.g., transcript) on next render
if "apply_after_render" not in st.session_state:
    st.session_state.apply_after_render = False # auto-apply the just-injected prompt (voice)
if "apply_source" not in st.session_state:
    st.session_state.apply_source = None

if "typed_submit" not in st.session_state:
    st.session_state.typed_submit = False       # enter/submit from keyboard
if "suppress_next_on_change" not in st.session_state:
    st.session_state.suppress_next_on_change = False  # avoid callback when we inject programmatically

if "last_audio_fp" not in st.session_state:
    st.session_state.last_audio_fp = None
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = None
if "last_extraction" not in st.session_state:
    st.session_state.last_extraction = None  # {"raw": "...", "payload": {...}, "source": "..."}

# ============================ CSS / LAYOUT ============================
sidebar_display = "block" if st.session_state.filters_open else "none"
st.markdown(f"""
<style>
/* Hide Streamlit default footer/menu */
#MainMenu, footer {{ visibility: hidden; }}

/* Sidebar collapsible */
[data-testid="stSidebar"] {{ display: {sidebar_display}; }}

/* Tighten spacing and leave extra room for bottom prompt bar */
.block-container {{ padding-top: 6px; padding-bottom: 132px; }}

/* Top bar */
.topbar {{
  position: sticky; top: 0; z-index: 100;
  background: #fff; border-bottom: 1px solid #eee;
  padding: 8px 10px; margin-bottom: 6px;
}}
.topbar .inner {{ display: flex; justify-content: space-between; align-items: center; }}
.topbar .title {{ font-weight: 600; font-size: 16px; }}
.topbar .btn {{
  background: #000; color: #fff; border: none; border-radius: 8px;
  padding: 6px 12px; font-weight: 600; cursor: pointer;
}}
.topbar .btn:hover {{ opacity: 0.9; }}

/* Bottom prompt bar (single bar) */
.prompt-wrap {{
  position: fixed; left: 24px; right: 24px; bottom: 20px; z-index: 1000;
}}
.prompt-inner {{ max-width: 1080px; margin: 0 auto; }}
.prompt-container {{ position: relative; }}

/* Make the text_input look like a clean pill */
.prompt-container input[type="text"] {{
  border-radius: 28px !important;
  padding-right: 56px !important; /* room for mic */
  height: 48px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}}

/* Mic button overlay (single icon, clearly visible) */
.mic-btn {{
  position: absolute;
  right: 8px;
  top: 50%;
  transform: translateY(-50%);
  width: 40px;
  height: 40px;
  border-radius: 999px;
  display: flex; align-items: center; justify-content: center;
  background: #ffffff;
  border: 1px solid #e5e5e5;
  box-shadow: 0 1px 6px rgba(0,0,0,0.08);
  z-index: 10;
}}
/* Optional subtle hover */
.mic-btn:hover {{ box-shadow: 0 2px 10px rgba(0,0,0,0.12); }}

/* When recording: slight red glow */
.mic-recording {{
  box-shadow: 0 0 0 3px rgba(255,0,0,0.15), 0 2px 10px rgba(255,0,0,0.25) !important;
  border-color: #ffb3b3 !important;
}}

/* tiny floating transcript preview */
.transient {{
  position: fixed; right: 24px; bottom: 80px; background: #111; color:#fff;
  padding: 8px 12px; border-radius: 10px; font-size: 12px; opacity: .92;
}}
</style>
""", unsafe_allow_html=True)

# ============================ TOP BAR ============================
st.markdown('<div class="topbar"><div class="inner">', unsafe_allow_html=True)
st.markdown('<div class="title">Scooter Wheels Scheduler</div>', unsafe_allow_html=True)
toggle_label = "Hide Filters" if st.session_state.filters_open else "Show Filters"
if st.button(toggle_label, key="toggle_filters_btn"):
    st.session_state.filters_open = not st.session_state.filters_open
    st.rerun()
st.markdown('</div></div>', unsafe_allow_html=True)

# ============================ SIDEBAR FILTERS =========================
if st.session_state.filters_open:
    with st.sidebar:
        st.header("Filters ⚙️")
        st.session_state.filt_max_orders = st.number_input(
            "Orders", 1, 100, value=st.session_state.filt_max_orders, step=1
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

        # ---- Debug panels ----
        with st.expander("🔎 Debug (voice & intent)"):
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

        with st.expander("🧰 Command log (recent)"):
            if st.session_state.cmd_log:
                st.dataframe(pd.DataFrame(st.session_state.cmd_log[-10:]), use_container_width=True)
            else:
                st.caption("No commands yet.")

# ============================ EFFECTIVE FILTERS =========================
max_orders = int(st.session_state.filt_max_orders)
wheel_choice = st.session_state.filt_wheels or sorted(base_schedule["wheel_type"].unique().tolist())
machine_choice = st.session_state.filt_machines or sorted(base_schedule["machine"].unique().tolist())

# ============================ ORDER NAME HELPERS =========================
# Accept "Order 5", "order five", "order #12", "Order twelve"
UNITS = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
    "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,
    "sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19
}
TENS = {"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90}

def words_to_int(s: str) -> int | None:
    s = s.lower().strip().replace("-", " ")
    if s in UNITS: return UNITS[s]
    if s in TENS: return TENS[s]
    # "twenty one"
    parts = s.split()
    if len(parts) == 2 and parts[0] in TENS and parts[1] in UNITS:
        return TENS[parts[0]] + UNITS[parts[1]]
    # digits
    if re.fullmatch(r"\d{1,4}", s):
        return int(s)
    return None

def normalize_order_name(text: str) -> str | None:
    """
    Extract order identifier from text and normalize to 'Order N'
    """
    m = re.search(r"\border\s*(?:#\s*)?([A-Za-z\-]+|\d{1,4})\b", text, flags=re.I)
    if not m:
        return None
    token = m.group(1)
    n = words_to_int(token)
    if n is None:  # maybe token like 'o012' sneaks in -> strip non-digits
        digits = re.findall(r"\d+", token)
        if digits:
            n = int(digits[-1])
    return f"Order {n}" if n is not None else None

# ============================ NLP / INTENT =========================
INTENT_SCHEMA = {
  "type": "object",
  "properties": {
    "intent": {"type": "string", "enum": ["delay_order", "move_order", "swap_orders"]},
    "order_id": {"type": "string"},          # now like "Order 12"
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

DEFAULT_TZ = "Asia/Makassar"
TZ = pytz.timezone(DEFAULT_TZ)

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
        if n is None: continue
        u = unit.lower()
        if u.startswith("d"): d["days"] += n
        elif u.startswith("h"): d["hours"] += n
        else: d["minutes"] += n
    return d

# ---------- Extraction via OpenAI (kept, but we sanitize IDs to 'Order N') ----------
def _extract_with_openai(user_text: str):
    from openai import OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    SYSTEM = (
        "You normalize factory scheduling edit commands for a Gantt. "
        "Return ONLY JSON matching the given schema. "
        "Supported intents: delay_order, move_order, swap_orders. "
        "Order IDs are of the form 'Order N' (e.g., 'Order 12'). "
        "If user says 'tomorrow' etc., convert to ISO date in Asia/Makassar. "
        "If time missing on move_order, default 08:00. "
        "If delay units missing, assume days; minutes allowed. "
        "If user mentions 'order five', resolve to 'Order 5'."
    )
    GUIDE = (
        '1) "delay order five one day" -> {"intent":"delay_order","order_id":"Order 5","days":1}\n'
        '2) "push order 9 by 1h 30m" -> {"intent":"delay_order","order_id":"Order 9","hours":1.0,"minutes":30.0}\n'
        '3) "move order twelve to Aug 30 09:13" -> {"intent":"move_order","order_id":"Order 12","date":"2025-08-30","time":"09:13"}\n'
        '4) "swap order 2 with order 3" -> {"intent":"swap_orders","order_id":"Order 2","order_id_2":"Order 3"}\n'
        '5) "advance order 8 by two days" -> {"intent":"delay_order","order_id":"Order 8","days":-2}\n'
    )
    resp = client.responses.create(
        model="gpt-5.1",
        input=[{"role":"system","content":SYSTEM},{"role":"user","content":GUIDE},{"role":"user","content":user_text}],
        response_format={"type":"json_schema","json_schema":{"name":"Edit","schema":INTENT_SCHEMA}}
    )
    text = resp.output[0].content[0].text
    data = json.loads(text)

    # sanitize order_id fields that may come back as "order five"
    for k in ("order_id", "order_id_2"):
        if k in data and isinstance(data[k], str):
            norm = normalize_order_name(data[k])
            if norm: data[k] = norm

    data["_source"] = "openai"
    return data

def _regex_fallback(user_text: str):
    t = user_text.strip()
    low = t.lower()

    # SWAP: "swap order 2 with order 3"
    m = re.search(r"(?:^|\b)(swap|switch)\s+(order[^,;]*)", low)
    if m:
        # try to find two order mentions
        ids = re.findall(r"\border\s*(?:#\s*)?([A-Za-z\-]+|\d{1,4})\b", low, flags=re.I)
        if len(ids) >= 2:
            a = normalize_order_name(f"order {ids[0]}")
            b = normalize_order_name(f"order {ids[1]}")
            if a and b and a != b:
                return {"intent": "swap_orders", "order_id": a, "order_id_2": b, "_source": "regex"}

    # DELAY (delay/push/postpone vs. advance/bring forward)
    delay_sign = +1
    if re.search(r"\b(advance|bring\s+forward|pull\s+in)\b", low):
        delay_sign = -1
        low_norm = re.sub(r"\b(advance|bring\s+forward|pull\s+in)\b", "delay", low)
    else:
        low_norm = low

    # Identify target order
    target = normalize_order_name(low_norm)

    # by <duration>
    m = re.search(r"\b(delay|push|postpone)\b.*?\bby\b\s+(.+)$", low_norm)
    if target and m:
        dur = _parse_duration_chunks(m.group(2))
        if any(v != 0 for v in dur.values()):
            return {
                "intent": "delay_order",
                "order_id": target,
                "days": delay_sign * dur["days"],
                "hours": delay_sign * dur["hours"],
                "minutes": delay_sign * dur["minutes"],
                "_source": "regex",
            }

    # implicit duration e.g. "delay order 7 two days"
    m = re.search(r"\b(delay|push|postpone)\b.*?(days?|d|hours?|h|minutes?|mins?|m)\b", low_norm)
    if target and m:
        dur = _parse_duration_chunks(low_norm)
        if any(v != 0 for v in dur.values()):
            return {
                "intent": "delay_order",
                "order_id": target,
                "days": delay_sign * dur["days"],
                "hours": delay_sign * dur["hours"],
                "minutes": delay_sign * dur["minutes"],
                "_source": "regex",
            }

    # MOVE: "move order 12 to/on <datetime>"
    m = re.search(r"\b(move|set|schedule)\b.*?\b(to|on)\s+(.+)", low)
    if m:
        target = target or normalize_order_name(low)
        when = m.group(3) if len(m.groups()) >= 3 else None
        if target and when:
            try:
                dt = dtp.parse(when, fuzzy=True)
                return {
                    "intent": "move_order",
                    "order_id": target,
                    "date": dt.date().isoformat(),
                    "time": dt.strftime("%H:%M"),
                    "_source": "regex",
                }
            except Exception:
                pass

    # simple fallback: "delay order one day"
    if target and re.search(r"\b(delay|push|postpone)\b.*\bone day\b", low):
        return {"intent": "delay_order", "order_id": target, "days": 1, "_source": "regex"}

    return {"intent": "unknown", "raw": user_text, "_source": "regex"}

def extract_intent(user_text: str) -> dict:
    try:
        if os.getenv("OPENAI_API_KEY"):
            return _extract_with_openai(user_text)
    except Exception:
        pass
    return _regex_fallback(user_text)

def validate_intent(payload: dict, orders_df, sched_df):
    intent = payload.get("intent")

    def order_exists(oid):
        return oid and (orders_df["order_id"] == oid).any()

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

# ============================ APPLY FUNCTIONS =========================
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

def apply_delay(schedule_df: pd.DataFrame, order_id: str, days=0, hours=0, minutes=0):
    s = schedule_df.copy()
    delta = timedelta(days=float(days or 0), hours=float(hours or 0), minutes=float(minutes or 0))
    mask = s["order_id"] == order_id
    s.loc[mask, "start"] = s.loc[mask, "start"] + delta
    s.loc[mask, "end"]   = s.loc[mask, "end"]   + delta
    return _repack_touched_machines(s, [order_id])

def apply_move(schedule_df: pd.DataFrame, order_id: str, target_dt):
    s = schedule_df.copy()
    t0 = s.loc[s["order_id"] == order_id, "start"].min()
    delta = target_dt - t0
    days = delta.days
    hours = delta.seconds // 3600
    minutes = (delta.seconds % 3600) // 60
    return apply_delay(s, order_id, days=days, hours=hours, minutes=minutes)

def apply_swap(schedule_df: pd.DataFrame, a: str, b: str):
    s = schedule_df.copy()
    a0 = s.loc[s["order_id"] == a, "start"].min()
    b0 = s.loc[s["order_id"] == b, "start"].min()
    da, db = (b0 - a0), (a0 - b0)
    s = apply_delay(s, a, days=da.days, hours=da.seconds // 3600, minutes=(da.seconds % 3600)//60)
    s = apply_delay(s, b, days=db.days, hours=db.seconds // 3600, minutes=(db.seconds % 3600)//60)
    return s

# ============================ GANTT =========================
sched = st.session_state.schedule_df.copy()
sched = sched[sched["wheel_type"].isin(wheel_choice)]
sched = sched[sched["machine"].isin(machine_choice)]
order_priority = sched.groupby("order_id", as_index=False)["start"].min().sort_values("start")
keep_ids = order_priority["order_id"].head(max_orders).tolist()
sched = sched[sched["order_id"].isin(keep_ids)].copy()

if sched.empty:
    st.info("No operations match the current filters.")
else:
    domain = ["Urban-200", "Offroad-250", "Racing-180", "HeavyDuty-300", "Eco-160"]
    range_ = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    select_order = alt.selection_point(fields=["order_id"], on="click", clear="dblclick")
    y_machines_sorted = sorted(sched["machine"].unique().tolist())
    base_enc = {
        "y": alt.Y("machine:N", sort=y_machines_sorted, title=None),
        "x": alt.X("start:T", title=None, axis=alt.Axis(format="%a %b %d")),
        "x2": "end:T",
    }
    bars = alt.Chart(sched).mark_bar(cornerRadius=3).encode(
        color=alt.condition(
            select_order,
            alt.Color("wheel_type:N", scale=alt.Scale(domain=domain, range=range_), legend=None),
            alt.value("#dcdcdc"),
        ),
        opacity=alt.condition(select_order, alt.value(1.0), alt.value(0.25)),
        tooltip=[
            alt.Tooltip("order_id:N", title="Order"),
            alt.Tooltip("operation:N", title="Operation"),
            alt.Tooltip("sequence:Q", title="Seq"),
            alt.Tooltip("machine:N", title="Machine"),
            alt.Tooltip("start:T", title="Start"),
            alt.Tooltip("end:T", title="End"),
            alt.Tooltip("due_date:T", title="Due"),
            alt.Tooltip("wheel_type:N", title="Wheel"),
        ],
    )
    labels = alt.Chart(sched).mark_text(
        align="left", dx=6, baseline="middle", fontSize=10, color="white"
    ).encode(
        text="order_id:N",
        opacity=alt.condition(select_order, alt.value(1.0), alt.value(0.7)),
    )
    gantt = (
        alt.layer(bars, labels, data=sched)
        .encode(**base_enc)
        .add_params(select_order)
        .properties(width="container", height=520)
        .configure_view(stroke=None)
    )
    st.altair_chart(gantt, use_container_width=True)

# ============================ DEEPGRAM (bytes) =========================
def _deepgram_transcribe_bytes(wav_bytes: bytes, mimetype: str = "audio/wav") -> str:
    key = os.getenv("DEEPGRAM_API_KEY")
    if not key:
        raise RuntimeError("DEEPGRAM_API_KEY not set")
    import requests
    url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&language=en"
    headers = {"Authorization": f"Token {key}", "Content-Type": mimetype}
    r = requests.post(url, headers=headers, data=wav_bytes, timeout=45)
    try:
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Deepgram error: {r.text}") from e
    j = r.json()
    try:
        return j["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
    except Exception:
        raise RuntimeError(f"Deepgram: no transcript in response: {j}")

# ============================ PIPELINE (shared) =========================
def _process_and_apply(cmd_text: str, *, source_hint: str = None):
    """
    Extract -> validate -> apply. Also records detailed logs for debugging.
    """
    from copy import deepcopy
    try:
        payload = extract_intent(cmd_text)

        # LOG what we sent and what we parsed
        st.session_state.last_extraction = {
            "raw": cmd_text,
            "payload": deepcopy(payload),
            "source": source_hint or payload.get("_source", "?"),
        }

        ok, msg = validate_intent(payload, orders, st.session_state.schedule_df)

        # Traditional rolling log (short)
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

# ============================ PROMPT BAR + INLINE MIC =========================
# Inject transcript BEFORE rendering the input
if st.session_state.pending_prompt is not None:
    st.session_state.suppress_next_on_change = True
    st.session_state.prompt_text = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    st.session_state.apply_after_render = True
    st.session_state.apply_source = "voice/deepgram"

def _on_prompt_change():
    if st.session_state.get("suppress_next_on_change"):
        st.session_state.suppress_next_on_change = False
        return
    st.session_state.typed_submit = True

# Single prompt bar (no duplicates)
with st.container():
    st.markdown('<div class="prompt-wrap"><div class="prompt-inner"><div class="prompt-container">', unsafe_allow_html=True)

    st.text_input(
        label="Prompt",
        key="prompt_text",
        label_visibility="collapsed",
        placeholder="Ask anything…",
        on_change=_on_prompt_change,
    )

    # Overlay mic icon (component renders in place; we wrap it for better visuals)
    mic_wrap = st.container()
    with mic_wrap:
        # We can add / remove class 'mic-recording' later if you want explicit state toggling
        st.markdown('<div class="mic-btn" id="mic-btn"></div>', unsafe_allow_html=True)
        rec = mic_recorder(
            start_prompt="", stop_prompt="",
            key="press_mic",
            just_once=False,
            format="wav",
            use_container_width=False
        )

    st.markdown('</div></div></div>', unsafe_allow_html=True)  # close prompt containers

# When user presses Enter (typed)
if st.session_state.typed_submit:
    st.session_state.typed_submit = False
    txt = (st.session_state.prompt_text or "").strip()
    if txt:
        _process_and_apply(txt, source_hint="text")

# When a recording finishes: transcribe -> store transcript to inject next render
if rec and isinstance(rec, dict) and rec.get("bytes"):
    wav_bytes = rec["bytes"]
    fp = (len(wav_bytes), hash(wav_bytes[:1024]))
    if fp != st.session_state.last_audio_fp:
        st.session_state.last_audio_fp = fp
        try:
            with st.spinner("Transcribing…"):
                transcript = _deepgram_transcribe_bytes(wav_bytes, mimetype="audio/wav")
            st.session_state.last_transcript = transcript  # exact text from Deepgram

            # Do not mutate the input now; set pending and rerun
            st.session_state.pending_prompt = transcript or ""
            if transcript:
                st.markdown(f'<div class="transient">🗣 {transcript}</div>', unsafe_allow_html=True)
            st.rerun()
        except Exception as e:
            st.error(f"Transcription failed: {e}")

# After render, if we injected a transcript, apply it automatically
if st.session_state.apply_after_render:
    st.session_state.apply_after_render = False
    to_send = (st.session_state.prompt_text or "").strip()
    if to_send:
        _process_and_apply(to_send, source_hint=st.session_state.apply_source or "voice/deepgram")
