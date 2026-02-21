# -*- coding: utf-8 -*-
"""
Експериментальне дослідження роботи методу забезпечення живучості інформаційної системи на мобільній платформі з урахуванням здійсненності резервних профілів ресурсозабезпечення

Скрипт генерує імітаційний сценарій:
- нестаціонарні ресурси платформи з критичними станами і частковим відновленням
- змінне некритичне навантаження, що конкурує з критичними процесами за ресурси
- переривчаста зв’язність з серійністю і залежністю від стану каналу
- неповні спостереження: за відсутності зв’язності контур керування ІСМП працює з застарілою оцінкою

Порівнюються три режими:
- proposed: здійсненність профілів + запас до межі здійсненності + кероване перемикання
- static: фіксований профіль
- naive: просте порогове перемикання без оцінювання здійсненності

Результати:
- out5/fig1_resources_connectivity.png
- out5/fig2_delta_compare.png
- out5/fig3_proposed_profile.png
- out5/metrics.csv і out5/timeseries.csv

Запуск:
python exp_section5_realistic.py

Залежності:
pip install numpy pandas matplotlib
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Cfg:
    seed: int = 21
    T: int = 600

    q: int = 3                  # ресурси
    nC: int = 3                 # критичні процеси

    dt: float = 1.0
    eps: float = 1e-9

    # обмеження реконфігурацій
    W: int = 40                 # довжина ковзного вікна
    L: int = 7                  # ліміт перемикань у вікні
    delta_keep: float = 0.08    # зона утримання

    w_delta: float = 1.0
    lambda_cost: float = 0.12

    # тривалість і накладні витрати реконфігурації
    reconf_steps: int = 4
    reconf_penalty: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.03, 0.02], dtype=float))  # штраф до доступних ресурсів

    # спостереження
    obs_noise: np.ndarray = field(default_factory=lambda: np.array([0.015, 0.02, 0.01], dtype=float))      # a_t=1
    stale_drift: float = 0.006                                 # дрейф оцінки при a_t=0

    # динаміка ресурсу як середньоповертаючий процес з критичними станами
    mu: np.ndarray = field(default_factory=lambda: np.array([0.65, 0.55, 0.80], dtype=float))              
    phi: np.ndarray = field(default_factory=lambda: np.array([0.985, 0.980, 0.995], dtype=float))          
    sigma: np.ndarray = field(default_factory=lambda: np.array([0.018, 0.025, 0.010], dtype=float))        
    shock_p: float = 0.035
    shock_scale: np.ndarray = field(default_factory=lambda: np.array([0.20, 0.25, 0.10], dtype=float))     
    recover_p: float = 0.020
    recover_scale: np.ndarray = field(default_factory=lambda: np.array([0.12, 0.15, 0.08], dtype=float))   

    # некритичне навантаження
    u_mu: np.ndarray = field(default_factory=lambda: np.array([0.22, 0.20, 0.10], dtype=float))
    u_phi: np.ndarray = field(default_factory=lambda: np.array([0.990, 0.990, 0.995], dtype=float))
    u_sigma: np.ndarray = field(default_factory=lambda: np.array([0.020, 0.020, 0.010], dtype=float))

    # переривчаста зв’язність: залежить від каналу
    p_on_base: float = 0.75
    p_on_gain: float = 0.35     # чим кращий канал, тим частіше a_t=1
    p_switch: float = 0.06      # серійність


@dataclass
class Profile:
    name: str
    W: np.ndarray           # shape (q, nC)
    cap: np.ndarray         # shape (q,)
    slack: np.ndarray       # shape (q,)
    nc_factor: np.ndarray   # shape (q,) коефіцієнт придушення некритичного навантаження
    cost: float             # базова ресурсна ціна профілю, використовується в матриці переходів


def make_profiles() -> List[Profile]:
    # профілі мають різні області здійсненності
    P_bal = Profile(
        name="Balanced",
        W=np.array([[0.40, 0.35, 0.25],
                    [0.35, 0.40, 0.25],
                    [0.38, 0.32, 0.30]], dtype=float),
        cap=np.array([0.80, 0.80, 0.78], dtype=float),
        slack=np.array([0.05, 0.05, 0.06], dtype=float),
        nc_factor=np.array([0.90, 0.90, 1.00], dtype=float),   
        cost=0.10,
    )
    P_energy = Profile(
        name="EnergySave",
        W=np.array([[0.42, 0.33, 0.25],
                    [0.34, 0.41, 0.25],
                    [0.45, 0.30, 0.25]], dtype=float),
        cap=np.array([0.82, 0.80, 0.70], dtype=float),
        slack=np.array([0.05, 0.05, 0.07], dtype=float),
        nc_factor=np.array([0.88, 0.88, 0.85], dtype=float),
        cost=0.14,
    )
    P_net = Profile(
        name="NetPriority",
        W=np.array([[0.36, 0.34, 0.30],
                    [0.48, 0.34, 0.18],
                    [0.36, 0.34, 0.30]], dtype=float),
        cap=np.array([0.78, 0.92, 0.78], dtype=float),
        slack=np.array([0.05, 0.04, 0.06], dtype=float),
        nc_factor=np.array([0.88, 0.78, 0.98], dtype=float),   
        cost=0.16,
    )
    P_surv = Profile(
        name="Survival",
        W=np.array([[0.46, 0.32, 0.22],
                    [0.40, 0.40, 0.20],
                    [0.46, 0.30, 0.24]], dtype=float),
        cap=np.array([0.96, 0.96, 0.92], dtype=float),
        slack=np.array([0.01, 0.01, 0.02], dtype=float),
        nc_factor=np.array([0.55, 0.55, 0.85], dtype=float),   
        cost=0.28,
    )
    return [P_bal, P_energy, P_net, P_surv]


def transition_cost_matrix(profiles: List[Profile]) -> np.ndarray:
    
    K = len(profiles)
    C = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            dW = np.linalg.norm(profiles[i].W - profiles[j].W)
            dcap = np.linalg.norm(profiles[i].cap - profiles[j].cap)
            dsl = np.linalg.norm(profiles[i].slack - profiles[j].slack)
            dnc = np.linalg.norm(profiles[i].nc_factor - profiles[j].nc_factor)
            C[i, j] = 0.55 * dW + 0.20 * dcap + 0.10 * dsl + 0.15 * dnc + 0.20 * (profiles[j].cost)
    mx = C.max() if C.max() > 0 else 1.0
    return C / mx


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.05, 1.0)


def generate_scenario(cfg: Cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)

    
    r_raw = np.zeros((cfg.T, cfg.q), dtype=float)
    u = np.zeros((cfg.T, cfg.q), dtype=float)
    a = np.zeros(cfg.T, dtype=int)

    r_raw[0] = np.array([0.80, 0.70, 0.90], dtype=float)
    u[0] = np.array([0.20, 0.18, 0.10], dtype=float)

    a[0] = 1
    for t in range(1, cfg.T):
        noise = rng.normal(0.0, cfg.sigma, size=cfg.q)
        r_raw[t] = cfg.mu + cfg.phi * (r_raw[t-1] - cfg.mu) + noise

        if rng.random() < cfg.shock_p:
            r_raw[t] -= cfg.shock_scale * rng.random(cfg.q)

        if rng.random() < cfg.recover_p:
            r_raw[t] += cfg.recover_scale * rng.random(cfg.q)

        r_raw[t] = clamp01(r_raw[t])

        u_noise = rng.normal(0.0, cfg.u_sigma, size=cfg.q)
        u[t] = cfg.u_mu + cfg.u_phi * (u[t-1] - cfg.u_mu) + u_noise
        u[t] = np.clip(u[t], 0.0, 0.70)

        if rng.random() < cfg.p_switch:
            p_on = np.clip(cfg.p_on_base + cfg.p_on_gain * (r_raw[t, 1] - cfg.mu[1]), 0.15, 0.95)
            a[t] = 1 if rng.random() < p_on else 0
        else:
            a[t] = a[t-1]

    return r_raw, u, a


def m_timevarying(cfg: Cfg) -> np.ndarray:
    M0 = np.array([[0.18, 0.10, 0.14],
                   [0.15, 0.14, 0.12],
                   [0.12, 0.09, 0.10]], dtype=float)
    rng = np.random.default_rng(cfg.seed + 300)
    # помірна випадкова варіація вимог, що відображає зміну інтенсивності критичної функції
    mult = 1.0 + rng.normal(0.0, 0.06, size=(cfg.T, cfg.nC, cfg.q))
    mult = np.clip(mult, 0.85, 1.20)
    return M0[None, :, :] * mult


def observe_resources(cfg: Cfg, r_eff: np.ndarray, a: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed + 500)
    r_hat = np.zeros_like(r_eff)
    r_hat[0] = r_eff[0]
    for t in range(1, cfg.T):
        if a[t] == 1:
            r_hat[t] = clamp01(r_eff[t] + rng.normal(0.0, cfg.obs_noise, size=cfg.q))
        else:
            drift = cfg.stale_drift * (0.5 + rng.random(cfg.q))
            r_hat[t] = clamp01(r_hat[t-1] - drift)
    return r_hat


def compute_allocation_minfirst(profile: Profile, r_avail: np.ndarray, M: np.ndarray, eps: float) -> np.ndarray:
    nC, q = M.shape
    x = np.zeros((nC, q), dtype=float)
    for j in range(q):
        budget = max(0.0, profile.cap[j] * r_avail[j] - profile.slack[j])
        min_need = float(np.sum(M[:, j]))
        give_min = min(min_need, budget)

        if min_need > eps:
            x[:, j] = M[:, j] * (give_min / min_need)

        remaining = budget - give_min
        if remaining > 0:
            w = np.maximum(profile.W[j, :], eps)
            w = w / w.sum()
            x[:, j] += remaining * w
    return x


def compute_delta(x: np.ndarray, r_avail: np.ndarray, M: np.ndarray, eps: float) -> float:
    crit = []
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            denom = max(M[i, j], eps)
            crit.append((x[i, j] - M[i, j]) / denom)
    delta_crit = float(np.min(crit))

    res = []
    for j in range(M.shape[1]):
        denom = max(r_avail[j], eps)
        res.append((r_avail[j] - float(np.sum(x[:, j]))) / denom)
    delta_res = float(np.min(res))
    return float(min(delta_crit, delta_res))


def window_switch_ok(switch_hist: List[int], W: int, L: int) -> bool:
    recent = switch_hist[max(0, len(switch_hist) - W):]
    return int(np.sum(recent)) < L


def simulate(cfg: Cfg) -> Tuple[pd.DataFrame, pd.DataFrame]:
    profiles = make_profiles()
    Ccost = transition_cost_matrix(profiles)
    K = len(profiles)

    r_raw, u_raw, a = generate_scenario(cfg)
    M_t = m_timevarying(cfg)

    regimes = ["proposed", "static", "naive"]
    k_prev = {reg: 0 for reg in regimes}
    sw_hist = {reg: [] for reg in regimes}
    reconf_left = {reg: 0 for reg in regimes}

    r_base = clamp01(r_raw - u_raw)

    r_hat = observe_resources(cfg, r_base, a)

    rows = []

    for t in range(cfg.T):
        for reg in regimes:
            if reg == "static":
                k = 0
            else:
                k = k_prev[reg]

            prof = profiles[k]

            u_eff = u_raw[t] * prof.nc_factor
            r_eff = clamp01(r_raw[t] - u_eff)

            if reconf_left[reg] > 0:
                r_eff = clamp01(r_eff - cfg.reconf_penalty)
                reconf_left[reg] -= 1

            r_seen = r_hat[t].copy()
            
            M = M_t[t]

            reason = "static"
            switched = 0

            if reg == "proposed":
                deltas_est = np.zeros(K, dtype=float)
                feasible = []
                for kk in range(K):
                    xk = compute_allocation_minfirst(profiles[kk], r_seen, M, cfg.eps)
                    deltas_est[kk] = compute_delta(xk, r_seen, M, cfg.eps)
                    if deltas_est[kk] >= 0.0:
                        feasible.append(kk)

                can_switch = (a[t] == 1) and window_switch_ok(sw_hist[reg], cfg.W, cfg.L)

                if (k_prev[reg] in feasible) and (deltas_est[k_prev[reg]] >= cfg.delta_keep):
                    k_new = k_prev[reg]
                    reason = "keep"
                else:
                    if can_switch and len(feasible) > 0:
                        best = None
                        for kk in feasible:
                            score = cfg.w_delta * deltas_est[kk] - cfg.lambda_cost * Ccost[k_prev[reg], kk]
                            if best is None or score > best[0]:
                                best = (score, kk)
                        k_new = int(best[1])
                        reason = "switch" if k_new != k_prev[reg] else "keep"
                    else:
                        k_new = k_prev[reg]
                        reason = "forced_keep"

                switched = 1 if (t > 0 and k_new != k_prev[reg]) else 0
                if switched == 1:
                    reconf_left[reg] = cfg.reconf_steps

                k_prev[reg] = k_new
                sw_hist[reg].append(switched)
                prof = profiles[k_new]
                u_eff = u_raw[t] * prof.nc_factor
                r_eff = clamp01(r_raw[t] - u_eff)
                if reconf_left[reg] > 0:
                    pass
                x = compute_allocation_minfirst(prof, r_eff, M, cfg.eps)
                delta = compute_delta(x, r_eff, M, cfg.eps)
                k = k_new

            elif reg == "naive":
                target = 0
                if r_seen[2] < 0.45:
                    target = 1
                elif r_seen[1] < 0.40:
                    target = 2
                elif r_seen[0] < 0.40:
                    target = 3

                can_switch = (a[t] == 1) and window_switch_ok(sw_hist[reg], cfg.W, cfg.L)
                if can_switch:
                    k_new = target
                    reason = "naive_switch" if k_new != k_prev[reg] else "naive_keep"
                else:
                    k_new = k_prev[reg]
                    reason = "forced_keep"

                switched = 1 if (t > 0 and k_new != k_prev[reg]) else 0
                if switched == 1:
                    reconf_left[reg] = cfg.reconf_steps

                k_prev[reg] = k_new
                sw_hist[reg].append(switched)
                prof = profiles[k_new]
                u_eff = u_raw[t] * prof.nc_factor
                r_eff = clamp01(r_raw[t] - u_eff)
                x = compute_allocation_minfirst(prof, r_eff, M, cfg.eps)
                delta = compute_delta(x, r_eff, M, cfg.eps)
                k = k_new

            else:
                sw_hist[reg].append(0)
                x = compute_allocation_minfirst(profiles[0], r_eff, M, cfg.eps)
                delta = compute_delta(x, r_eff, M, cfg.eps)

            rows.append({
                "t": t,
                "regime": reg,
                "a": int(a[t]),
                "r_cpu": float(r_eff[0]),
                "r_bw": float(r_eff[1]),
                "r_energy": float(r_eff[2]),
                "k": int(k),
                "profile": profiles[k].name,
                "switched": int(switched),
                "delta": float(delta),
                "reason": reason,
            })

    df = pd.DataFrame(rows)

    # метрики
    mets = []
    for reg in regimes:
        d = df[df["regime"] == reg].sort_values("t")
        T = len(d)
        delta = d["delta"].values
        sw = d["switched"].values
        a_arr = d["a"].values

        J_feas = float(np.mean(delta >= 0.0))
        J_delta = float(np.mean(delta))
        J_def = float(np.mean(np.maximum(0.0, -delta)))
        J_sw = int(np.sum(sw))
        J_conn = float(np.sum(sw * a_arr) / (float(np.sum(sw)) + cfg.eps))
        idx = np.where(delta < 0.0)[0]
        J_ttf = int(idx[0]) if len(idx) > 0 else T

        mets.append({
            "regime": reg,
            "J_feas": J_feas,
            "J_delta": J_delta,
            "J_def": J_def,
            "J_sw": J_sw,
            "J_conn": J_conn,
            "J_ttf": J_ttf,
        })

    metrics = pd.DataFrame(mets)
    return df, metrics


def save_plots(df: pd.DataFrame, metrics: pd.DataFrame, out_dir: str):
    """
    3 фігури
    """
    os.makedirs(out_dir, exist_ok=True)

    base = df[df["regime"] == "proposed"].sort_values("t")
    t = base["t"].values

    # -------------------------
    # Фігура 1: ресурси і зв’язність
    # -------------------------
    fig, ax1 = plt.subplots(figsize=(12.5, 4.8))

    l_cpu, = ax1.plot(t, base["r_cpu"].values, label="CPU", color="tab:blue", linewidth=1.6)
    l_bw,  = ax1.plot(t, base["r_bw"].values, label="Канал", color="tab:orange", linewidth=1.6)
    l_en,  = ax1.plot(t, base["r_energy"].values, label="Енергія", color="tab:green", linewidth=1.6)

    ax1.set_xlabel("t")
    ax1.set_ylabel("нормований рівень ресурсу")
    ax1.set_title("Сценарій ресурсів платформи та переривчастої зв’язності")

    ax2 = ax1.twinx()
    l_a, = ax2.step(
        t, base["a"].values, where="post",
        label="зв’язність a_t", color="tab:red", linestyle="--", linewidth=1.6
    )
    ax2.set_ylabel("a_t")
    ax2.set_ylim(-0.05, 1.05)

    handles = [l_cpu, l_bw, l_en, l_a]
    labels = [h.get_label() for h in handles]
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=4, frameon=False
    )
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(os.path.join(out_dir, "fig1_resources_connectivity.png"), dpi=260, bbox_inches="tight")
    plt.close(fig)

    # -------------------------
    # Фігура 2: delta(t) порівняння режимів
    # -------------------------
    fig, ax = plt.subplots(figsize=(12.5, 4.8))

    series = [
        ("proposed", "здійсненність + запас", "tab:blue"),
        ("static",   "фіксований профіль",    "tab:orange"),
        ("naive",    "порогове перемикання",  "tab:green"),
    ]
    lines = []
    for reg, name, col in series:
        d = df[df["regime"] == reg].sort_values("t")
        ln, = ax.plot(d["t"].values, d["delta"].values, label=name, color=col, linewidth=1.6)
        lines.append(ln)

    ax.axhline(0.0, linewidth=1.2, color="black")
    ax.set_xlabel("t")
    ax.set_ylabel("delta")
    ax.set_title("Запас до межі здійсненності у часі")

    fig.legend(
        lines, [ln.get_label() for ln in lines],
        loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=3, frameon=False
    )
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(os.path.join(out_dir, "fig2_delta_compare.png"), dpi=260, bbox_inches="tight")
    plt.close(fig)

    # -------------------------
    # Фігура 3: обраний профіль у proposed + провали зв’язності
    # -------------------------
    fig, ax = plt.subplots(figsize=(12.5, 4.8))

    l_prof, = ax.step(t, base["k"].values, where="post", label="індекс профілю", color="tab:blue", linewidth=1.6)

    sw = base["switched"].values
    sw_idx = np.where(sw == 1)[0]
    l_sw = ax.scatter(
        t[sw_idx], base["k"].values[sw_idx],
        marker="o", s=24, label="перемикання", color="tab:orange"
    )

    # затінення інтервалів a_t=0
    a = base["a"].values
    l_shade = None
    if np.any(a == 0):
        shade = ax.fill_between(
            t, -0.6, float(base["k"].max()) + 0.6,
            where=(a == 0), step="post", alpha=0.12,
            label="a_t = 0"
        )
        l_shade = shade

    ax.set_xlabel("t")
    ax.set_ylabel("профіль")
    ax.set_title("Кероване перемикання резервних профілів у запропонованому методі")

    yt = list(range(int(base["k"].max()) + 1))
    ax.set_yticks(yt)
    mp = {int(r["k"]): r["profile"] for _, r in base[["k", "profile"]].drop_duplicates().iterrows()}
    ax.set_yticklabels([mp.get(i, str(i)) for i in yt])

    handles = [l_prof, l_sw]
    if l_shade is not None:
        handles.append(l_shade)
    fig.legend(
        handles, [h.get_label() for h in handles],
        loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=len(handles), frameon=False
    )
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(os.path.join(out_dir, "fig3_proposed_profile.png"), dpi=260, bbox_inches="tight")
    plt.close(fig)

    # таблиця метрик
    metrics.to_csv(os.path.join(out_dir, "metrics.csv"), index=False, encoding="utf-8")


def main():
    cfg = Cfg()
    df, metrics = simulate(cfg)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "out5")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "timeseries.csv"), index=False, encoding="utf-8")
    save_plots(df, metrics, out_dir)

    print("Готово. Папка результатів:", out_dir)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
