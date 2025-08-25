import os, json, re
from datetime import timedelta
import pytz
from dateutil import parser as dtp

import streamlit as st
import pandas as pd
import altair as alt
from streamlit_mic_recorder import mic_recorder

# -------------------- PAGE / KEYS --------------------
st.set_page_config(page_title="Scooter Wheels Scheduler", layout="wide")
for k in ("OPENAI_API_KEY", "DEEPGRAM_API_KEY"):
    try:
        os.environ[k] = os.environ.get(k) or st.secrets[k]
    except Exception:
        pass

# -------------------- DATA --------------------
def _oNNN_to_order(text: str) -> str:
    if isinstance(text, str):
        m = re.fullmatch(r"[Oo](\d{1,4})", text.strip())
        if m: return f"Order {int(m.group(1))}"
    return text

def _normalize_loaded_ids(df: pd.DataFrame, col="order_id"):
    if col in df.columns: df[col] = df[col].apply(_oNNN_to_order)
    return df

@st.cache_data
def load_data():
    orders = pd.read_csv("data/scooter_orders.csv", parse_dates=["due_date"])
    sched  = pd.read_csv("data/scooter_schedule.csv", parse_dates=["start","end","due_date"])
    return _normalize_loaded_ids(orders), _normalize_loaded_ids(sched)

orders, base_schedule = load_data()
st.session_state.setdefault("schedule_df", base_schedule.copy())
st.session_state.setdefault("filt_max_orders", 20)
st.session_state.setdefault("filt_wheels", sorted(base_schedule["wheel_type"].unique().tolist()))
st.session_state.setdefault("filt_machines", sorted(base_schedule["machine"].unique().tolist()))
st.session_state.setdefault("last_audio_fp", None)
st.session_state.setdefault("last_transcript", None)
st.session_state.setdefault("last_extraction", None)

# -------------------- CSS (SAFE) --------------------
st.markdown("""
<style>
#MainMenu, footer {display:none;}
/* Leave space for fixed bottom bar */
.block-container { padding-bottom: 170px !important; }

/* Bottom fixed bar (about ~15% height on common screens) */
.bottom-wrap{
  position:fixed; left:0; right:0; bottom:0; z-index:1000;
  background:linear-gradient(to top, rgba(255,255,255,0.98), rgba(255,255,255,0.96));
  border-top:1px solid #eee;
  padding:12px 16px 16px 16px;
}
.bottom-inner{
  max-width:1100px; margin:0 auto;
  display:grid; grid-template-columns:3fr 1fr; gap:16px; align-items:center;
}

/* Transcript */
.transcript-box{
  min-height:54px; border:1px dashed #e2e2e2; border-radius:12px;
  background:#fbfbfb; padding:12px 14px; font-size:14px; color:#222;
  display:flex; align-items:center; overflow:hidden; text-overflow:ellipsis;
}
.placeholder{color:#888}

/* Mic (style the recorder widget itself) */
.mic-area{display:grid; place-items:center;}
.rec-wrap{ position:relative; width:72px; height:72px; }
.rec-wrap > div{ width:72px !important; height:72px !important; }
.rec-wrap button, .rec-wrap [role="button"]{
  width:72px !important; height:72px !important; padding:0 !important;
  border-radius:999px !important; cursor:pointer;
  background:radial-gradient(ellipse at 30% 30%, #fff 0%, #f6f6f6 60%, #eee 100%) !important;
  border:1px solid #e6e6e6 !important; box-shadow:0 6px 18px rgba(0,0,0,.12) !important;
}
.rec-wrap button:hover, .rec-wrap [role="button"]:hover{ transform:translateY(-1px); box-shadow:0 8px 22px rgba(0,0,0,.16) !important; }
.rec-wrap button:active, .rec-wrap [role="button"]:active{ transform:scale(.98); }
/* Visible mic glyph */
.rec-wrap button::before, .rec-wrap [role="button"]::before{
  content:""; display:block; width:30px; height:30px; margin:0 auto;
  background:#111;
  -webkit-mask:url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 14 0h-2zm-5 8a1 1 0 1 0 0-2 1 1 0 0 0 0 2zm-1 1h2v3h-2v-3z"/></svg>') center/contain no-repeat;
          mask:url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 14 0h-2zm-5 8a1 1 0 1 0 0-2 1 1 0 0 0 0 2zm-1 1h2v3h-2v-3z"/></svg>') center/contain no-repeat;
}
/* Hide any stray checkbox/labels the component may render */
.rec-wrap input[type="checkbox"], .rec-wrap label{ display:none !important; }
</style>
""", unsafe_allow_html=True)

# -------------------- FILTERS (optional) --------------------
with st.expander("Filters", expanded=False):
    st.session_state["filt_max_orders"] = st.number_input("Orders to show", 1, 100, value=st.session_state["filt_max_orders"])
    wheels_all = sorted(base_schedule["wheel_type"].unique().tolist())
    st.session_state["filt_wheels"] = st.multiselect("Wheel", wheels_all, default=st.session_state["filt_wheels"] or wheels_all)
    machines_all = sorted(base_schedule["machine"].unique().tolist())
    st.session_state["filt_machines"] = st.multiselect("Machine", machines_all, default=st.session_state["filt_machines"] or machines_all)

# -------------------- GANTT --------------------
sched = st.session_state["schedule_df"].copy()
sched = sched[sched["wheel_type"].isin(st.session_state["filt_wheels"] or sorted(base_schedule["wheel_type"].unique().tolist()))]
sched = sched[sched["machine"].isin(st.session_state["filt_machines"] or sorted(base_schedule["machine"].unique().tolist()))]
order_priority = sched.groupby("order_id", as_index=False)["start"].min().sort_values("start")
keep_ids = order_priority["order_id"].head(int(st.session_state["filt_max_orders"])).tolist()
sched = sched[sched["order_id"].isin(keep_ids)].copy()

st.subheader("Production Schedule")
if sched.empty:
    st.info("No operations match the current filters.")
else:
    domain = ["Urban-200","Offroad-250","Racing-180","HeavyDuty-300","Eco-160"]
    range_ = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]
    select_order = alt.selection_point(fields=["order_id"], on="click", clear="dblclick")
    y_sorted = sorted(sched["machine"].unique().tolist())
    base_enc = {
        "y": alt.Y("machine:N", sort=y_sorted, title=None),
        "x": alt.X("start:T", title=None, axis=alt.Axis(format="%a %b %d")),
        "x2": "end:T",
    }
    bars = alt.Chart(sched).mark_bar(cornerRadius=3).encode(
        color=alt.condition(select_order,
            alt.Color("wheel_type:N", scale=alt.Scale(domain=domain, range=range_), legend=None),
            alt.value("#dcdcdc")),
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
    ).encode(text="order_id:N", opacity=alt.condition(select_order, alt.value(1.0), alt.value(0.7)))
    gantt = (
        alt.layer(bars, labels, data=sched)
        .encode(**base_enc)
        .add_params(select_order)
        .properties(width="container", height=560)  # visible, reliable
        .configure_view(stroke=None)
    )
    st.altair_chart(gantt, use_container_width=True)

# -------------------- DEEPGRAM --------------------
def _deepgram_transcribe_bytes(wav_bytes: bytes, mimetype: str = "audio/wav") -> str:
    key = os.getenv("DEEPGRAM_API_KEY")
    if not key: raise RuntimeError("DEEPGRAM_API_KEY not set")
    import requests
    url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&language=en"
    headers = {"Authorization": f"Token {key}", "Content-Type": mimetype}
    r = requests.post(url, headers=headers, data=wav_bytes, timeout=45)
    r.raise_for_status()
    j = r.json()
    return j["results"]["channels"][0]["alternatives"][0]["transcript"].strip()

# -------------------- NLP HELPERS --------------------
UNITS = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
         "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,
         "sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19}
TENS = {"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90}

def words_to_int(s:str)->int|None:
    s=s.lower().strip().replace("-"," ")
    if s in UNITS: return UNITS[s]
    if s in TENS: return TENS[s]
    parts=s.split()
    if len(parts)==2 and parts[0] in TENS and parts[1] in UNITS: return TENS[parts[0]]+UNITS[parts[1]]
    if re.fullmatch(r"\d{1,4}", s): return int(s)
    return None

def normalize_order_name(text:str)->str|None:
    if not isinstance(text,str): return None
    legacy=re.search(r"\bO(\d{1,4})\b", text, flags=re.I)
    if legacy: return f"Order {int(legacy.group(1))}"
    m=re.search(r"\border\s*(?:#\s*)?([A-Za-z\-]+|\d{1,4})\b", text, flags=re.I)
    if not m: return None
    token=m.group(1)
    n=words_to_int(token)
    if n is None:
        digits=re.findall(r"\d+", token)
        if digits: n=int(digits[-1])
    return f"Order {n}" if n is not None else None

INTENT_SCHEMA={"type":"object","properties":{
    "intent":{"type":"string","enum":["delay_order","move_order","swap_orders"]},
    "order_id":{"type":"string"},"order_id_2":{"type":"string"},
    "days":{"type":"number"},"hours":{"type":"number"},"minutes":{"type":"number"},
    "date":{"type":"string"},"time":{"type":"string"},
    "timezone":{"type":"string","default":"Asia/Makassar"},"note":{"type":"string"}},
    "required":["intent"],"additionalProperties":False}

DEFAULT_TZ="Asia/Makassar"
TZ=pytz.timezone(DEFAULT_TZ)

def _parse_duration_chunks(text:str):
    d={"days":0.0,"hours":0.0,"minutes":0.0}
    def numtok(tok):
        tok=tok.strip().lower().replace(",",".").replace("-"," ")
        try: return float(tok)
        except: pass
        v=words_to_int(tok)
        return float(v) if v is not None else None
    for num,unit in re.findall(r"([\d\.,]+|\b[\w\-]+\b)\s*(days?|d|hours?|h|minutes?|mins?|m)\b", text, flags=re.I):
        n=numtok(num); 
        if n is None: continue
        u=unit.lower()
        if u.startswith("d"): d["days"]+=n
        elif u.startswith("h"): d["hours"]+=n
        else: d["minutes"]+=n
    return d

def _extract_with_openai(user_text:str):
    from openai import OpenAI
    if not os.getenv("OPENAI_API_KEY"): raise RuntimeError("OPENAI_API_KEY not set")
    client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    SYSTEM=("You normalize factory scheduling edit commands for a Gantt. Return ONLY JSON matching the schema. "
            "Intents: delay_order, move_order, swap_orders. Order IDs are 'Order N'. "
            "Convert relative dates to Asia/Makassar; default move time 08:00.")
    GUIDE=('1) "delay order five one day" -> {"intent":"delay_order","order_id":"Order 5","days":1}\n'
           '2) "push order 9 by 1h 30m" -> {"intent":"delay_order","order_id":"Order 9","hours":1.0,"minutes":30.0}\n'
           '3) "move order twelve to Aug 30 09:13" -> {"intent":"move_order","order_id":"Order 12","date":"2025-08-30","time":"09:13"}\n'
           '4) "swap order 2 with order 3" -> {"intent":"swap_orders","order_id":"Order 2","order_id_2":"Order 3"}')
    resp=client.responses.create(
        model="gpt-5.1",
        input=[{"role":"system","content":SYSTEM},{"role":"user","content":GUIDE},{"role":"user","content":user_text}],
        response_format={"type":"json_schema","json_schema":{"name":"Edit","schema":INTENT_SCHEMA}}
    )
    text=resp.output[0].content[0].text
    data=json.loads(text)
    for k in ("order_id","order_id_2"):
        if k in data and isinstance(data[k], str):
            norm=normalize_order_name(data[k])
            if norm: data[k]=norm
    data["_source"]="openai"
    return data

def _regex_fallback(user_text:str):
    t=user_text.strip(); low=t.lower()
    m=re.search(r"(?:^|\b)(swap|switch)\s+(order[^,;]*)", low)
    if m:
        ids=re.findall(r"\border\s*(?:#\s*)?([A-Za-z\-]+|\d{1,4})\b", low, flags=re.I)
        legacy=re.findall(r"\bO(\d{1,4})\b", low, flags=re.I)
        tokens=ids+legacy
        if len(tokens)>=2:
            a=normalize_order_name(f"order {tokens[0]}"); b=normalize_order_name(f"order {tokens[1]}")
            if a and b and a!=b: return {"intent":"swap_orders","order_id":a,"order_id_2":b,"_source":"regex"}
    delay_sign=+1
    if re.search(r"\b(advance|bring\s+forward|pull\s+in)\b", low):
        delay_sign=-1; low=re.sub(r"\b(advance|bring\s+forward|pull\s+in)\b","delay", low)
    target=normalize_order_name(low) or normalize_order_name(t)
    m=re.search(r"\b(delay|push|postpone)\b.*?\bby\b\s+(.+)$", low)
    if target and m:
        dur=_parse_duration_chunks(m.group(2))
        if any(v!=0 for v in dur.values()):
            return {"intent":"delay_order","order_id":target,"days":delay_sign*dur["days"],"hours":delay_sign*dur["hours"],"minutes":delay_sign*dur["minutes"],"_source":"regex"}
    m=re.search(r"\b(delay|push|postpone)\b.*?(days?|d|hours?|h|minutes?|mins?|m)\b", low)
    if target and m:
        dur=_parse_duration_chunks(low)
        if any(v!=0 for v in dur.values()):
            return {"intent":"delay_order","order_id":target,"days":delay_sign*dur["days"],"hours":delay_sign*dur["hours"],"minutes":delay_sign*dur["minutes"],"_source":"regex"}
    m=re.search(r"\b(move|set|schedule)\b.*?\b(to|on)\s+(.+)", low)
    if m:
        target=target or normalize_order_name(low); when=m.group(3)
        if target and when:
            try:
                dt=dtp.parse(when, fuzzy=True)
                return {"intent":"move_order","order_id":target,"date":dt.date().isoformat(),"time":dt.strftime("%H:%M"),"_source":"regex"}
            except: pass
    if target and re.search(r"\b(delay|push|postpone)\b.*\bone day\b", low):
        return {"intent":"delay_order","order_id":target,"days":1,"_source":"regex"}
    return {"intent":"unknown","raw":t,"_source":"regex"}

def extract_intent(user_text:str)->dict:
    try:
        if os.getenv("OPENAI_API_KEY"): return _extract_with_openai(user_text)
    except Exception: pass
    return _regex_fallback(user_text)

DEFAULT_TZ="Asia/Makassar"; TZ=pytz.timezone(DEFAULT_TZ)
def validate_intent(payload:dict, orders_df, sched_df):
    intent=payload.get("intent")
    def exists(oid): return oid and (orders_df["order_id"]==oid).any()
    if intent not in ("delay_order","move_order","swap_orders"): return False, "Unsupported intent"
    if intent in ("delay_order","move_order","swap_orders"):
        oid=payload.get("order_id")
        if not exists(oid): return False, f"Unknown order_id: {oid}"
    if intent=="swap_orders":
        oid2=payload.get("order_id_2")
        if not exists(oid2): return False, f"Unknown order_id_2: {oid2}"
        if oid2==payload.get("order_id"): return False, "Cannot swap the same order."
        return True,"ok"
    if intent=="delay_order":
        for k in ("days","hours","minutes"):
            if k in payload and payload[k] is not None:
                try: payload[k]=float(payload[k])
                except: return False, f"{k.capitalize()} must be numeric."
        if not any(payload.get(k) for k in ("days","hours","minutes")):
            return False,"Delay needs a duration (days/hours/minutes)."
        return True,"ok"
    if intent=="move_order":
        date_str=payload.get("date"); hhmm=payload.get("time") or "08:00"
        if not date_str: return False, "Move needs a date."
        try:
            dt=dtp.parse(f"{date_str} {hhmm}")
            if dt.tzinfo is None: dt=TZ.localize(dt)
            else: dt=dt.astimezone(TZ)
            payload["_target_dt"]=dt
        except: return False, f"Unparseable datetime: {date_str} {hhmm}"
        return True,"ok"
    return False,"Invalid payload"

def _repack_touched_machines(s:pd.DataFrame,touched):
    machines=s.loc[s["order_id"].isin(touched),"machine"].unique().tolist()
    for m in machines:
        idx=s.index[s["machine"]==m]
        block=s.loc[idx].sort_values(["start","end"]).copy()
        last=None
        for i,row in block.iterrows():
            if last is not None and row["start"]<last:
                dur=row["end"]-row["start"]
                s.at[i,"start"]=last
                s.at[i,"end"]=last+dur
            last=s.at[i,"end"]
    return s

def apply_delay(s:pd.DataFrame, oid:str, days=0,hours=0,minutes=0):
    s=s.copy(); d=timedelta(days=float(days or 0),hours=float(hours or 0),minutes=float(minutes or 0))
    mask=s["order_id"]==oid
    s.loc[mask,"start"]=s.loc[mask,"start"]+d
    s.loc[mask,"end"]=s.loc[mask,"end"]+d
    return _repack_touched_machines(s,[oid])

def apply_move(s:pd.DataFrame, oid:str, target_dt):
    t0=s.loc[s["order_id"]==oid,"start"].min()
    delta=target_dt-t0
    return apply_delay(s, oid, days=delta.days, hours=delta.seconds//3600, minutes=(delta.seconds%3600)//60)

def apply_swap(s:pd.DataFrame, a:str, b:str):
    s=s.copy()
    a0=s.loc[s["order_id"]==a,"start"].min()
    b0=s.loc[s["order_id"]==b,"start"].min()
    da,db=(b0-a0),(a0-b0)
    s=apply_delay(s, a, days=da.days, hours=da.seconds//3600, minutes=(da.seconds%3600)//60)
    s=apply_delay(s, b, days=db.days, hours=db.seconds//3600, minutes=(db.seconds%3600)//60)
    return s

# -------------------- FIXED BOTTOM BAR --------------------
st.markdown('<div class="bottom-wrap"><div class="bottom-inner">', unsafe_allow_html=True)

# Transcript (75%)
if st.session_state["last_transcript"]:
    st.markdown(f'<div class="transcript-box">{st.session_state["last_transcript"]}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="transcript-box placeholder">(your transcript will appear here)</div>', unsafe_allow_html=True)

# Mic (25%) – single recorder
st.markdown('<div class="mic-area"><div class="rec-wrap">', unsafe_allow_html=True)
rec = mic_recorder(start_prompt="", stop_prompt="", key="press_mic", just_once=False, format="wav", use_container_width=False)
st.markdown('</div></div>', unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)

# -------------------- RECORD → TRANSCRIBE → APPLY --------------------
def _deepgram_transcribe_bytes(wav_bytes: bytes, mimetype="audio/wav")->str:
    key=os.getenv("DEEPGRAM_API_KEY")
    if not key: raise RuntimeError("DEEPGRAM_API_KEY not set")
    import requests
    url="https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&language=en"
    headers={"Authorization":f"Token {key}","Content-Type":mimetype}
    r=requests.post(url, headers=headers, data=wav_bytes, timeout=45)
    r.raise_for_status(); j=r.json()
    return j["results"]["channels"][0]["alternatives"][0]["transcript"].strip()

def extract_and_apply(transcript:str):
    payload=extract_intent(transcript)
    st.session_state["last_extraction"]={"raw":transcript,"payload":payload,"source":payload.get("_source","?")}
    ok,msg=validate_intent(payload, orders, st.session_state["schedule_df"])
    if not ok:
        st.error(f"❌ Cannot apply: {msg}")
        return
    if payload["intent"]=="delay_order":
        st.session_state["schedule_df"]=apply_delay(
            st.session_state["schedule_df"], payload["order_id"],
            days=payload.get("days",0), hours=payload.get("hours",0), minutes=payload.get("minutes",0))
        st.success(f"✅ Delayed {payload['order_id']}.")
    elif payload["intent"]=="move_order":
        st.session_state["schedule_df"]=apply_move(st.session_state["schedule_df"], payload["order_id"], payload["_target_dt"])
        st.success(f"✅ Moved {payload['order_id']}.")
    elif payload["intent"]=="swap_orders":
        st.session_state["schedule_df"]=apply_swap(st.session_state["schedule_df"], payload["order_id"], payload["order_id_2"])
        st.success(f"✅ Swapped {payload['order_id']} and {payload['order_id_2']}.")
    st.rerun()

if rec and isinstance(rec, dict) and rec.get("bytes"):
    wav=rec["bytes"]
    fp=(len(wav), hash(wav[:1024]))
    if fp != st.session_state["last_audio_fp"]:
        st.session_state["last_audio_fp"]=fp
        try:
            with st.spinner("Transcribing…"):
                txt=_deepgram_transcribe_bytes(wav)
            st.session_state["last_transcript"]=txt
            if txt: extract_and_apply(txt)
            else: st.warning("No speech detected.")
        except Exception as e:
            st.error(f"Transcription failed: {e}")
