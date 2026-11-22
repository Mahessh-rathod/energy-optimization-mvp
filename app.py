import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ai_rag_engine as rag

st.set_page_config(page_title="Energy Optimization MVP", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@st.cache_data
def load_data():
    fc = pd.read_csv(os.path.join(DATA_DIR, "forecast_10d.csv"), parse_dates=["Datetime"])
    hist = pd.read_csv(os.path.join(DATA_DIR, "recent_history.csv"), parse_dates=["Datetime"])
    with open(os.path.join(DATA_DIR, "forecast_summary.json")) as f:
        summary = json.load(f)
    return fc, hist, summary


@st.cache_resource
def get_index():
    return rag.DocIndex()


try:
    fc, hist, summary = load_data()
except FileNotFoundError:
    st.error("Forecast files not found. Run the forecast cell (Cell 5) first.")
    st.stop()

index = get_index()


def provider_label():
    return rag.last_provider if rag.last_provider != "none" else "offline fallback (AI busy)"


# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write("Forecast, optimization, AI insights and a RAG chatbot for energy demand.")
    st.caption(f"Data source: {summary.get('data_source', 'unknown')}")
    st.caption(f"Forecast: {summary['forecast_start'][:16]} to {summary['forecast_end'][:16]}")
    st.caption("AI chain: Gemini, then Groq, then offline fallback")

def has_secret(name):
    try:
        return name in st.secrets
    except Exception:
        return False


with st.sidebar:
    with st.expander("AI diagnostics"):
        st.write("Gemini secret found:", has_secret("My_api_key"))
        st.write("Groq secret found:", has_secret("Groq_api_key"))
        if st.button("Test AI providers"):
            for name, fn in (("Gemini", rag._ask_gemini), ("Groq", rag._ask_groq)):
                try:
                    fn("Reply with the single word: OK", rag.SYSTEM_PROMPT)
                    st.success(f"{name}: working")
                except Exception as exc:
                    st.error(f"{name}: {type(exc).__name__}: {str(exc)[:200]}")
# ---------- Header and KPIs ----------
st.title("Energy Optimization Dashboard")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Peak load", f"{summary['peak_load_mw']:,.0f} MW")
k2.metric("Average load", f"{summary['avg_predicted_mw']:,.0f} MW")
k3.metric("Validation MAPE", f"{summary['validation_mape_percent']:.2f}%")
k4.metric("Simulated saving", f"{summary['estimated_mwh_saved']:,.0f} MWh")

tabs = st.tabs(["Forecast", "Optimization", "AI insights", "Chatbot"])

# ---------- Module 1: Forecast ----------
with tabs[0]:
    st.subheader("Next 10 days: demand forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["Datetime"], y=hist["Energy_MW"],
                             name="Actual (last 7 days)", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=fc["Datetime"], y=fc["Predicted_MW"],
                             name="Forecast (next 240 hours)", line=dict(width=2, dash="dot")))
    fig.update_layout(height=420, xaxis_title="Time", yaxis_title="Load (MW)",
                      legend=dict(orientation="h", y=1.12), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig)

    st.subheader("Daily breakdown")
    daily = pd.DataFrame.from_dict(summary["daily_breakdown"], orient="index")
    daily.index.name = "Date"
    daily = daily.rename(columns={"mean": "Average MW", "max": "Peak MW", "min": "Lowest MW"})
    st.dataframe(daily)
    st.download_button("Download forecast CSV", fc.to_csv(index=False), "forecast_10d.csv", "text/csv")

# ---------- Module 2: Optimization ----------
with tabs[1]:
    st.subheader("Peak shaving and load shifting (simulation)")
    base_peak = fc["Predicted_MW"].max()
    opt_peak = fc["Optimized_MW"].max()
    o1, o2, o3 = st.columns(3)
    o1.metric("Baseline peak", f"{base_peak:,.0f} MW")
    o2.metric("Optimized peak", f"{opt_peak:,.0f} MW",
              f"{(opt_peak - base_peak) / base_peak * 100:.1f}%", delta_color="inverse")
    o3.metric("Simulated energy saved", f"{summary['estimated_mwh_saved']:,.0f} MWh")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=fc["Datetime"], y=fc["Predicted_MW"], name="Baseline"))
    fig2.add_trace(go.Scatter(x=fc["Datetime"], y=fc["Optimized_MW"], name="Optimized"))
    fig2.update_layout(height=380, xaxis_title="Time", yaxis_title="Load (MW)",
                       legend=dict(orientation="h", y=1.12), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2)

    st.subheader("Average daily profile by hour")
    prof = fc.assign(Hour=fc["Datetime"].dt.hour).groupby("Hour")[["Predicted_MW", "Optimized_MW"]].mean()
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=prof.index, y=prof["Predicted_MW"], name="Baseline"))
    fig3.add_trace(go.Bar(x=prof.index, y=prof["Optimized_MW"], name="Optimized"))
    fig3.update_layout(barmode="group", height=360, xaxis_title="Hour of day", yaxis_title="Average load (MW)",
                       legend=dict(orientation="h", y=1.12), margin=dict(l=10, r=10, t=30, b=10))
    fig3.update_yaxes(range=[prof.min().min() * 0.9, prof.max().max() * 1.02])
    st.plotly_chart(fig3)

    st.caption("Simulation assumptions: 12% of load is shaved in hours 14 to 20, and half of the shaved "
               "energy is moved to hours 1 to 5. These are assumptions, not measured savings.")

# ---------- Module 3: AI insights ----------
with tabs[2]:
    st.subheader("AI insights")
    st.write("Generates a short report from the forecast numbers and your documents.")
    if st.button("Generate AI insights"):
        with st.spinner("Asking the AI..."):
            st.session_state["insights"] = rag.generate_insights(summary, index)
            st.session_state["insights_by"] = provider_label()
    if "insights" in st.session_state:
        st.markdown(st.session_state["insights"])
        st.caption(f"Answered by: {st.session_state['insights_by']}")

# ---------- Module 4: RAG chatbot ----------
with tabs[3]:
    st.subheader("Ask about the forecast or your documents")
    st.caption("Try: 'When is the peak?', 'What is the peak tariff?', 'What do we do on low-demand days?'")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = st.chat_input("Ask a question...")
    if question:
        history = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, hits = rag.answer_question(question, summary, index, history)
            st.markdown(answer)
            st.caption(f"Answered by: {provider_label()}")
            if hits:
                with st.expander("Sources used"):
                    for src, chunk, score in hits:
                        st.markdown(f"**{src}** (match score {score:.2f})")
                        st.write(chunk)
        st.session_state.messages.append({"role": "assistant", "content": answer})
