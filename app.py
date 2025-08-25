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
.block-container {{ padding-top: 6px; padding-bottom: 124px; }}

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
.prompt-inner {{
  position: relative;
  max-width: 1080px; margin: 0 auto;
}}
/* The actual input sits inside this container, we overlay the mic on the right */
.prompt-container {{
  position: relative;
}}
/* Make the text_input look like a pill */
.prompt-container input[type="text"] {{
  border-radius: 28px !important;
  padding-right: 52px !important; /* room for mic */
  height: 44px;
}}

/* Mic button overlay (single icon) */
.mic-btn {{
  position: absolute;
  right: 8px;
  top: 50%;
  transform: translateY(-50%);
  width: 36px;
  height: 36px;
  border-radius: 999px;
  display: flex; align-items: center; justify-content: center;
  background: #f3f3f3;
  border: 1px solid #e5e5e5;
  z-index: 10;
}}
/* When recording we make the background red-ish; the component itself shows the icon */
.mic-recording {{
  background: #ffebeb !important;
  border-color: #ffb3b3 !important;
}}

/* tiny floating transcript preview */
.transient {{
  position: fixed; right: 24px; bottom: 74px; background: #111; color:#fff;
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


# ============================ NLP / INTENT =========================
INTENT_SCHEMA = {
  "type": "object",
  "properties": {
    "intent": {"type": "string", "enum": ["delay_order", "move_order", "swap_orders"]},
    "order_id": {"type": "string", "pattern": "^O\\d{3}$"},
    "order_id_2": {"type": "string", "pattern": "^O\\d{3}$"},
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

NUM_WORDS = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,
    "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
    "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,
    "sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,"twenty":20
}
def _num_token_to_float(tok: str):
    t = tok.strip().lower().replace("-", " ").replace(",", ".")
    try:
        return float(t)
    except Exception:
        pass
    parts = [p for p in t.split() if p]
    if len(parts) == 1 and parts[0] in NUM_WORDS:
        return float(NUM_WORDS[parts[0]])
    if len(parts) == 2 and parts[0] in NUM_WORDS and parts[1] in NUM_WORDS:
        return float(NUM_WORDS[parts[0]] + NUM_WORDS[parts[1]])
    return None

def _parse_duration_chunks(text: str):
    d = {"days":0.0,"hours":0.0,"minutes":0.0}
    for num, unit in re.findall(r"([\d\.,]+|\b\w+\b)\s*(days?|d|hours?|h|minutes?|mins?|m)\b", text, flags=re.I):
        n = _num_token_to_float(num)
        if n is None: continue
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
        "You normalize factory scheduling edit commands for a Gantt. "
        "Return ONLY JSON matching the given schema. "
        "Supported intents: delay_order, move_order, swap_orders. "
        "Order IDs look like O021 (3 digits). "
        "If user says 'tomorrow' etc., convert to ISO date in Asia/Makassar. "
        "If time missing on move_order, default 08:00. "
        "If delay units missing, assume days; minutes allowed."
    )
    USER_GUIDE = (
        '1) "delay O021 one day" -> {"intent":"delay_order","order_id":"O021","days":1}\n'
        '2) "push order O009 by 1h 30m" -> {"intent":"delay_order","order_id":"O009","hours":1.0,"minutes":30.0}\n'
        '3) "move o014 to Aug 30 09:13" -> {"intent":"move_order","order_id":"O014","date":"2025-08-30","time":"09:13"}\n'
        '4) "swap o027 with o031" -> {"intent":"swap_orders","order_id":"O027","order_id_2":"O031"}\n'
        '5) "advance O008 by two days" -> {"intent":"delay_order","order_id":"O008","days":-2}\n'
    )
    resp = client.responses.create(
        model="gpt-5.1",
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_GUIDE},
            {"role": "user", "content": user_text},
        ],
        response_format={"type": "json_schema", "json_schema": {"name": "Edit", "schema": INTENT_SCHEMA}},
    )
    text = resp.output[0].content[0].text
    data = json.loads(text)
    data["_source"] = "openai"
    return data

def _regex_fallback(user_text: str):
    t = user_text.strip()
    low = t.lower()

    # SWAP
    m = re.search(r"(?:^|\b)(swap|switch)\s+(o\d{3})\s*(?:with|and|&)?\s*(o\d{3})\b", low)
    if m:
        return {"intent":"swap_orders","order_id":m.group(2).upper(),"order_id_2":m.group(3).upper(),"_source":"regex"}

    # DELAY +/- (advance = negative)
    delay_sign = +1
    if re.search(r"\b(advance|bring\s+forward|pull\s+in)\b", low):
        delay_sign = -1
        low_norm = re.sub(r"\b(advance|bring\s+forward|pull\s+in)\b", "delay", low)
    else:
        low_norm = low

    m = re.search(r"(delay|push|postpone)\s+(o\d{3}).*?\bby\b\s+(.+)$", low_norm)
    if m:
        oid = m.group(2).upper()
        dur = _parse_duration_chunks(m.group(3))
        if any(v!=0 for v in dur.values()):
            return {"intent":"delay_order","order_id":oid,
                    "days":delay_sign*dur["days"],"hours":delay_sign*dur["hours"],
                    "minutes":delay_sign*dur["minutes"],"_source":"regex"}

    m = re.search(r"(delay|push|postpone)\s+(o\d{3}).*?(days?|d|hours?|h|minutes?|mins?|m)\b", low_norm)
    if m:
        oid = m.group(2).upper()
        dur = _parse_duration_chunks(low_norm)
        if any(v!=0 for v in dur.values()):
            return {"intent":"delay_order","order_id":oid,
                    "days":delay_sign*dur["days"],"hours":delay_sign*dur["hours"],
                    "minutes":delay_sign*dur["minutes"],"_source":"regex"}

    m = re.search(r"(move|set|schedule)\s+(o\d{3})\s+(to|on)\s+(.+)", low)
    if m:
        oid = m.group(2).upper(); when = m.group(4)
        try:
            dt = dtp.parse(when, fuzzy=True)
            return {"intent":"move_order","order_id":oid,"date":dt.date().isoformat(),"time":dt.strftime("%H:%M"),"_source":"regex"}
        except Exception:
            pass

    m = re.search(r"(delay|push|postpone)\s+(o\d{3}).*\b(one)\s+day\b", low)
    if m: return {"intent":"delay_order","order_id":m.group(2).upper(),"days":1,"_source":"regex"}
    return {"intent":"unknown","raw":user_text,"_source":"regex"}

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
        tooltip=["order_id","operation","sequence","machine","start","end","due_date","wheel_type"],
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
# Inject transcript BEFORE rendering the input, so we won't mutate widget state after instantiation
if st.session_state.pending_prompt is not None:
    st.session_state.suppress_next_on_change = True  # don't treat this as user typing
    st.session_state.prompt_text = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    # mark to auto-apply (voice) after we render this run
    st.session_state.apply_after_render = True

# Handle typing: when user presses Enter, on_change fires and we set typed_submit=True
def _on_prompt_change():
    if st.session_state.get("suppress_next_on_change"):
        # Swallow the change that came from us injecting the transcript
        st.session_state.suppress_next_on_change = False
        return
    st.session_state.typed_submit = True

# Single prompt bar (no duplicates)
prompt_area = st.container()
with prompt_area:
    st.markdown('<div class="prompt-wrap"><div class="prompt-inner">', unsafe_allow_html=True)
    st.markdown('<div class="prompt-container">', unsafe_allow_html=True)

    # Our single text input (looks like a chat bar). Using session_state["prompt_text"] for value.
    st.text_input(
        label="Prompt",
        key="prompt_text",
        label_visibility="collapsed",
        placeholder="Ask anything…",
        on_change=_on_prompt_change,      # Enter = submit
    )

    # Overlay mic icon (single icon component). It appears inside the input on the right.
    mic_box = st.container()
    with mic_box:
        st.markdown(
            f'<div class="mic-btn {"mic-recording" if st.session_state.get("mic_is_recording") else ""}"></div>',
            unsafe_allow_html=True
        )
        # The component returns a dict when recording ends (on release)
        rec = mic_recorder(
            start_prompt="", stop_prompt="",
            key="press_mic",
            just_once=False,
            format="wav",
            use_container_width=False
        )
    st.markdown('</div>', unsafe_allow_html=True)   # close prompt-container
    st.markdown('</div></div>', unsafe_allow_html=True)  # close prompt-wrap/inner

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

            # 1) DO NOT mutate prompt_text now (widget already rendered). Instead:
            st.session_state.pending_prompt = transcript or ""

            # 2) Show small preview
            if transcript:
                st.markdown(f'<div class="transient">🗣 {transcript}</div>', unsafe_allow_html=True)

            # Trigger a rerun. Next run will inject transcript before rendering the widget.
            st.rerun()
        except Exception as e:
            st.error(f"Transcription failed: {e}")

# After render (this run), if we had injected a transcript, apply it automatically
if st.session_state.apply_after_render:
    st.session_state.apply_after_render = False
    to_send = (st.session_state.prompt_text or "").strip()
    if to_send:
        _process_and_apply(to_send, source_hint=st.session_state.apply_source or "voice/deepgram")
