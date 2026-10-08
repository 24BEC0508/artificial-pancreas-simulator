import time
from dataclasses import dataclass
from typing import List

import numpy as np
import streamlit as st
import pandas as pd

from ap.params import Patient, Meal, TARGET, HYPO, HYPER, DEFAULT_PATIENT
from ap.model import equilibrium, rk4_step
from ap.controllers import PIDController, ConstrainedPI, LQRObserverController
from ap import tuned


@dataclass
class MealEntry:
    name: str
    quantity_g: float
    calories_kcal: float
    carbs_g: float
    sugar_g: float
    protein_g: float = 0.0
    fat_g: float = 0.0
    time_min: float = 0.0


class MealSchedule:
    """Adapter that lets the existing Bergman model accept multiple custom meals."""

    def __init__(self, entries: List[MealEntry]):
        self.entries = entries
        self.t_meal = min((m.time_min for m in entries), default=0.0)

    def D(self, t, patient: Patient):
        total = 0.0
        for m in self.entries:
            total += Meal(carbs_g=m.carbs_g, t_meal=m.time_min).D(t, patient)
        return total


def build_controller(kind, patient, ts, target):
    if kind == "PID":
        return PIDController(patient, **tuned.PID, Ts=ts, r=target)
    if kind == "Constrained PI (IOB)":
        return ConstrainedPI(patient, **tuned.PI, Ts=ts, r=target)
    return LQRObserverController(patient, **tuned.LQR, Ts=ts, r=target)


def init_state(patient, controller_kind, target, ts, noise_sd, delay_min, meals):
    controller = build_controller(controller_kind, patient, ts, target)
    controller.reset()
    x = equilibrium(patient)
    n_delay = max(0, int(round(delay_min / ts)))
    return {
        "t": 0.0,
        "x": x,
        "controller": controller,
        "history_g": [float(x[0])] * (n_delay + 1),
        "meal_schedule": MealSchedule(meals),
        "ts": ts,
        "dt": min(0.5, ts),
        "noise_sd": noise_sd,
        "delay_min": delay_min,
        "rng": np.random.default_rng(0),
        "t_data": [0.0],
        "g_data": [float(x[0])],
        "u_data": [float(patient.u_basal)],
        "i_data": [float(patient.Ib)],
        "iob_data": [0.0],
        "meal_data": [0.0],
        "y_data": [float(x[0])],
        "finished": False,
    }


def step_sim(state, patient):
    ts = state["ts"]
    dt = state["dt"]
    n_sub = max(1, int(round(ts / dt)))
    t = state["t"]
    x = state["x"]

    delay_steps = max(0, int(round(state["delay_min"] / ts)))
    state["history_g"].append(float(x[0]))
    idx = max(0, len(state["history_g"]) - 1 - delay_steps)
    y_true = state["history_g"][idx]
    y = y_true + (
        state["rng"].normal(0.0, state["noise_sd"])
        if state["noise_sd"] > 0
        else 0.0
    )

    u = float(state["controller"](t, y))
    meal = state["meal_schedule"]

    for _ in range(n_sub):
        state["t_data"].append(t)
        state["g_data"].append(float(x[0]))
        state["u_data"].append(u)
        state["i_data"].append(float(x[2]))
        state["meal_data"].append(float(meal.D(t, patient)))
        state["y_data"].append(float(y))
        x = rk4_step(t, x, u, dt, patient, meal)
        t += dt

    state["t"] = t
    state["x"] = x

    # For PI this is its exact internal IOB state.
    # For PID/LQR it is a useful excess-insulin estimate from plasma insulin.
    if hasattr(state["controller"], "B"):
        iob = max(0.0, float(state["controller"].B))
    else:
        iob = max(0.0, float(patient.VI * (x[2] - patient.Ib)))
    state["iob_data"].append(iob)

    return {
        "glucose": float(x[0]),
        "insulin": float(u),
        "plasma_insulin": float(x[2]),
        "iob": iob,
        "measurement": float(y),
        "time": float(t),
    }


def reset_sim():
    st.session_state.pop("sim", None)


st.set_page_config(
    page_title="Artificial Pancreas Simulator",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Artificial Pancreas Simulator")
st.caption("Bergman minimal model • PID • constrained PI + IOB • LQR + observer")

if "meals" not in st.session_state:
    st.session_state.meals = []

with st.sidebar:
    st.header("⚙️ Simulation settings")

    controller_kind = st.selectbox(
        "Controller",
        ["Constrained PI (IOB)", "PID", "LQR + observer"],
    )
    target = st.number_input(
        "Glucose target (mg/dL)", 70.0, 180.0, float(TARGET), step=5.0
    )
    duration = st.number_input(
        "Simulation duration (min)", 30, 1440, 240, step=30
    )
    ts = st.number_input(
        "CGM/controller period (min)", 1.0, 15.0, 5.0, step=1.0
    )
    noise_sd = st.number_input(
        "CGM noise SD (mg/dL)", 0.0, 20.0, 3.0, step=1.0
    )
    delay_min = st.number_input(
        "CGM delay (min)", 0.0, 30.0, 10.0, step=5.0
    )

    with st.expander("Advanced patient parameters"):
        gb = st.number_input("Basal glucose Gb (mg/dL)", 60.0, 160.0, float(DEFAULT_PATIENT.Gb))
        ib = st.number_input("Basal insulin Ib (uU/mL)", 1.0, 30.0, float(DEFAULT_PATIENT.Ib))
        p1 = st.number_input("p1 (1/min)", 0.001, 0.2, float(DEFAULT_PATIENT.p1), format="%.5f")
        p2 = st.number_input("p2 (1/min)", 0.001, 0.2, float(DEFAULT_PATIENT.p2), format="%.5f")
        p3 = st.number_input("p3", 1e-7, 1e-3, float(DEFAULT_PATIENT.p3), format="%.7g")
        n = st.number_input("Insulin clearance n (1/min)", 0.01, 0.5, float(DEFAULT_PATIENT.n), format="%.4f")

    patient = Patient(
        p1=p1, p2=p2, p3=p3, n=n, Gb=gb, Ib=ib,
        VI=DEFAULT_PATIENT.VI, VG=DEFAULT_PATIENT.VG
    )

    st.divider()
    st.write(f"**Basal pump rate:** {patient.u_basal:.2f} mU/min")
    st.write("**Pump limit:** 200 mU/min")

st.subheader("🍽️ Meal input")

with st.form("meal_form", clear_on_submit=True):
    c1, c2, c3 = st.columns(3)
    food = c1.text_input("Food / meal name", placeholder="e.g. Chicken biryani")
    quantity = c2.number_input("Quantity (g)", min_value=0.0, value=250.0, step=10.0)
    meal_time = c3.number_input(
        "Meal time (simulation min)", min_value=0.0,
        max_value=float(duration), value=30.0, step=5.0
    )

    c4, c5, c6 = st.columns(3)
    calories = c4.number_input("Calories (kcal)", min_value=0.0, value=500.0, step=10.0)
    carbs = c5.number_input("Carbohydrates (g)", min_value=0.0, value=60.0, step=1.0)
    sugar = c6.number_input("Sugar (g)", min_value=0.0, value=10.0, step=1.0)

    c7, c8 = st.columns(2)
    protein = c7.number_input("Protein (g)", min_value=0.0, value=20.0, step=1.0)
    fat = c8.number_input("Fat (g)", min_value=0.0, value=15.0, step=1.0)

    if st.form_submit_button("➕ Add meal"):
        if not food.strip():
            st.error("Give the meal a name.")
        elif carbs <= 0:
            st.error(
                "Carbohydrates must be greater than 0 g because carbohydrate grams "
                "drive the meal disturbance in the Bergman model."
            )
        else:
            st.session_state.meals.append(
                MealEntry(
                    name=food.strip(),
                    quantity_g=quantity,
                    calories_kcal=calories,
                    carbs_g=carbs,
                    sugar_g=sugar,
                    protein_g=protein,
                    fat_g=fat,
                    time_min=meal_time,
                )
            )
            reset_sim()
            st.success(
                f"Added {food.strip()} — {carbs:.0f} g carbs at t={meal_time:.0f} min."
            )

if st.session_state.meals:
    st.subheader("Planned meals")
    rows = [
        {
            "Time (min)": m.time_min,
            "Food": m.name,
            "Quantity (g)": m.quantity_g,
            "Calories (kcal)": m.calories_kcal,
            "Carbs (g)": m.carbs_g,
            "Sugar (g)": m.sugar_g,
        }
        for m in sorted(st.session_state.meals, key=lambda z: z.time_min)
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    if c1.button("🗑️ Clear meals", use_container_width=True):
        st.session_state.meals = []
        reset_sim()
        st.rerun()
    if c2.button("↻ Reset simulation", use_container_width=True):
        reset_sim()
        st.rerun()
else:
    st.info("Add at least one meal before starting the simulation.")

# Any change to model/controller settings starts a fresh simulation.
config_signature = (
    controller_kind, float(target), float(ts), float(noise_sd), float(delay_min),
    float(gb), float(ib), float(p1), float(p2), float(p3), float(n), int(duration),
)
if st.session_state.get("config_signature") != config_signature:
    reset_sim()
    st.session_state.config_signature = config_signature


def ensure_sim():
    if "sim" not in st.session_state:
        st.session_state.sim = init_state(
            patient, controller_kind, target, float(ts),
            float(noise_sd), float(delay_min), st.session_state.meals
        )
    return st.session_state.sim


if st.session_state.meals:
    c1, c2 = st.columns(2)
    step_once = c1.button("⏭️ Step one CGM sample", use_container_width=True)
    run_live = c2.button(
        "▶️ Run live simulation",
        type="primary",
        use_container_width=True,
    )

    if step_once:
        sim = ensure_sim()
        if sim["t"] < float(duration):
            step_sim(sim, patient)
            st.rerun()

    if run_live:
        sim = ensure_sim()
        chart_placeholder = st.empty()
        metric_placeholder = st.empty()
        progress_placeholder = st.empty()

        while sim["t"] < float(duration):
            out = step_sim(sim, patient)

            with metric_placeholder.container():
                m1, m2, m3, m4 = st.columns(4)
                g = out["glucose"]
                status = "LOW" if g < HYPO else "HIGH" if g > HYPER else "IN RANGE"
                m1.metric("Glucose", f"{g:.1f} mg/dL", status)
                m2.metric("Insulin delivery", f"{out['insulin']:.2f} mU/min")
                m3.metric("Plasma insulin", f"{out['plasma_insulin']:.2f} uU/mL")
                m4.metric("Insulin on board", f"{out['iob']:.1f} mU")

            chart_df = pd.DataFrame(
                {
                    "Glucose": sim["g_data"],
                    "Target": [target] * len(sim["g_data"]),
                },
                index=sim["t_data"],
            )
            chart_df.index.name = "Simulation time (min)"
            chart_placeholder.line_chart(chart_df, height=380)

            progress_placeholder.progress(
                min(1.0, sim["t"] / float(duration)),
                text=f"Simulated {sim['t']:.0f} / {duration:.0f} min",
            )

            time.sleep(0.20)

        sim["finished"] = True
        st.success("Simulation complete. Use Reset simulation to start again.")

if "sim" in st.session_state and len(st.session_state.sim["t_data"]) > 1:
    sim = st.session_state.sim

    st.subheader("📊 Simulation results")

    glucose_df = pd.DataFrame(
        {
            "Glucose (mg/dL)": sim["g_data"],
            "Target (mg/dL)": [target] * len(sim["g_data"]),
        },
        index=sim["t_data"],
    )
    glucose_df.index.name = "Simulation time (min)"

    insulin_df = pd.DataFrame(
        {
            "Insulin delivery (mU/min)": sim["u_data"],
            "Plasma insulin (uU/mL)": sim["i_data"],
        },
        index=sim["t_data"],
    )
    insulin_df.index.name = "Simulation time (min)"

    iob_df = pd.DataFrame(
        {"Insulin on board (mU)": sim["iob_data"]},
        index=np.linspace(0, sim["t"], len(sim["iob_data"])),
    )
    iob_df.index.name = "Simulation time (min)"

    st.line_chart(glucose_df, height=350)
    st.line_chart(insulin_df, height=300)
    st.line_chart(iob_df, height=250)

    g_now = sim["g_data"][-1]
    if g_now < HYPO:
        st.error(f"⚠️ Low glucose: {g_now:.1f} mg/dL")
    elif g_now > HYPER:
        st.warning(f"⚠️ High glucose: {g_now:.1f} mg/dL")
    else:
        st.success(f"🟢 Glucose in target range: {g_now:.1f} mg/dL")

st.divider()
st.caption(
    "Coursework simulator only. Parameters are typical literature values, not patient data. "
    "Calories and sugar are recorded as meal metadata; carbohydrate grams drive the "
    "glucose-appearance disturbance in the Bergman model."
)
