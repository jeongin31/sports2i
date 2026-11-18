"""초구 예측 공통 모듈: 데이터·정답 정의, 채점 함수, 결과 파일 저장/불러오기

모든 모델 노트북은 여기 함수로 데이터를 만들고 채점해야 서로 비교가 맞음
(first_pitch_baseline.ipynb의 함수를 그대로 옮긴 것, 베이스라인 노트북은 기록으로 그대로 둠)

결과 폴더 규칙: results/<모델명>/
    per_pitch.parquet   공별 예측 (save_results로 저장, compare.ipynb가 자동으로 읽음)
    *.csv               그 모델의 요약 표 (자유)

per_pitch.parquet 열 규칙 (2019 검증 + 2020 평가 초구 전부, 한 공 한 행)
    필수  gameID, pit_no                       공 구분
          p_type_<구종 9개>                    구종 확률 (CLASSES 순서, 합 1)
          p_zone_<24칸>                        실전 24칸 확률 (예측 구종으로 섞은 것, 합 1)
    선택  ll_joint                             구종×칸 log loss (공별)
          ll_zone_true                         실제 구종을 알 때 24칸 log loss
          dist, nll_coord                      실전 좌표 거리(cm)·NLL
    선택 열이 없으면 그 지표만 비교에서 빠짐
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.special import log_softmax, ndtr, ndtri  # scipy.optimize는 이 PC에서 차단(_trlib DLL)되어 쓰지 않음

# ---------------------------------------------------------------- 상수
VAL_YEAR, TEST_YEAR = 2019, 2020
BALL, HALF_W = 7.4, 21.6  # 공 하나 cm, 홈플레이트 반폭 cm
MERGE = {}  # 구종 합치기 (없음)
CLASSES = ["CHUP", "CURV", "CUTT", "FAST", "FORK", "KNUC", "SINK", "SLID", "TWOS"]
HSC = ["H", "S", "C"]
RESULTS_DIR = "results"
KEYS = ["gameID", "pit_no"]

# ---------------------------------------------------------------- 24칸 위치 정의 (타자 기준, + = 몸쪽)
X_EDGES = np.array([-HALF_W - BALL, -HALF_W + BALL, 0, HALF_W - BALL, HALF_W + BALL])  # 바깥 → 몸쪽


def z_edges(top, bot):  # 높이 경계 (공마다 그 타자 존 기준)
    return np.column_stack([bot - BALL, bot + BALL, (top + bot) / 2, top - BALL, top + BALL])


def cell_name(zi, xi):
    side = "몸" if xi >= 3 else "밖"
    if zi in (0, 5) or xi in (0, 5):
        return "C_" + ("높" if zi == 5 else "낮" if zi == 0 else "") + (side if xi in (0, 5) else "")
    row = {4: "상띠", 3: "상", 2: "하", 1: "하띠"}[zi]
    col = {1: "밖띠", 2: "밖", 3: "몸", 4: "몸띠"}[xi]
    return ("H_" + row + col) if zi in (2, 3) and xi in (2, 3) else ("S_" + row + "-" + col)


CELL_NAMES = np.array([cell_name(zi, xi) for zi in range(6) for xi in range(6)])  # zi*6 + xi
ZONES = list(dict.fromkeys(CELL_NAMES))
CELL2ZONE = (CELL_NAMES[:, None] == np.array(ZONES)[None, :]).astype(float)
ZONE_GROUP = np.array([z[0] for z in ZONES])
GROUP2 = (ZONE_GROUP[:, None] == np.array(HSC)[None, :]).astype(float)  # 24칸 → H/S/C
assert len(ZONES) == 24

TYPE_IDX = {c: i for i, c in enumerate(CLASSES)}
ZONE_IDX = {z: i for i, z in enumerate(ZONES)}
P_TYPE = ["p_type_" + c for c in CLASSES]
P_ZONE = ["p_zone_" + z for z in ZONES]

# ---------------------------------------------------------------- 위치 모멘트·보정 구간
MOM = ["x", "zn", "xx", "zz", "xz"]  # 위치 모멘트 (좌우, 높이 비율, 제곱, 곱)
BINS = np.linspace(0, 1, 11)  # 보정 10% 구간


# ---------------------------------------------------------------- 데이터
def load_data(data_dir=""):
    """전체 투구 m (위치 칸·높이 비율 포함)과 투수 이름 반환"""
    m = pd.read_parquet(data_dir + "pitch_model.parquet")
    m = m.sort_values(KEYS).reset_index(drop=True)
    m["game_date"] = pd.to_datetime(m["gameID"].str[:8], format="%Y%m%d")
    m["pitch_type"] = m["pitch_stuff"].replace(MERGE)
    m["same_side"] = (m["p_throws"] == m["STANCE"]).astype(int)  # 투수 손 = 타자 스탠스
    for c in ["TOP_SZ_CM", "BOTTOM_SZ_CM"]:
        m[c] = m[c].round(1)  # 존 높이 부동소수점 오차 제거
    m["label_doubt"] = m["flag_speed_gap"] | m["flag_fast_neg_z"]  # 구종 표시 의심
    assert sorted(m["pitch_type"].unique()) == CLASSES, "구종 목록이 CLASSES와 다름"

    x, z = m["plate_x_in_CM"].to_numpy(), m["CROSS_PLATE_Z_CM"].to_numpy()
    top, bot = m["TOP_SZ_CM"].to_numpy(), m["BOTTOM_SZ_CM"].to_numpy()
    xi = np.digitize(x, X_EDGES)
    zi = (z[:, None] >= z_edges(top, bot)).sum(1)
    m["cell"] = zi * 6 + xi
    m["zone"] = CELL_NAMES[m["cell"]]
    m["zone_group"] = m["zone"].str[0]
    m["z_norm"] = (z - bot) / (top - bot)  # 0 = 존 아랫선, 1 = 윗선
    out = np.maximum.reduce([np.abs(x) - (HALF_W + BALL), z - (top + BALL), (bot - BALL) - z])
    m["attack"] = np.select([m["zone_group"].eq("H"), m["zone_group"].eq("S"), out <= 2 * BALL],
                            ["Heart", "Shadow", "Chase"], "Waste")  # 분석용

    names = (pd.read_parquet(data_dir + "pitch_clean_full.parquet", columns=["pitcher_code", "pitcher_name"])
             .drop_duplicates().set_index("pitcher_code")["pitcher_name"])
    return m, names


def first_pitches(m):
    """초구(0-0, 구종 의심 제외) fp와 정답 번호, 검증·평가 행 표시 반환"""
    fp = m[m["ball_before"].eq(0) & m["strike_before"].eq(0) & ~m["label_doubt"]].reset_index(drop=True)
    fp["zn"] = fp["z_norm"]
    fp["xx"], fp["zz"], fp["xz"] = fp["plate_x_in_CM"] ** 2, fp["zn"] ** 2, fp["plate_x_in_CM"] * fp["zn"]
    fp = fp.rename(columns={"plate_x_in_CM": "x"})
    fp = pd.concat([fp,
                    pd.get_dummies(fp["pitch_type"]).reindex(columns=CLASSES, fill_value=0).astype(float),
                    pd.get_dummies(fp["zone"]).reindex(columns=ZONES, fill_value=0).astype(float)], axis=1)
    y_type = fp["pitch_type"].map(TYPE_IDX).to_numpy()
    y_zone = fp["zone"].map(ZONE_IDX).to_numpy()
    val = fp["GYEAR"].eq(VAL_YEAR).to_numpy()
    test = fp["GYEAR"].eq(TEST_YEAR).to_numpy()
    return fp, y_type, y_zone, val, test


# ---------------------------------------------------------------- 전날까지 누적
_TABLES = {}


def prior_table(data, keys, cols, half_life=None, weight=None, season_f=1.0, cache=True):
    """그룹·경기 날짜별 누적 합계 S, 개수 N (half_life: 공 N개마다 가중치 절반)
    weight: 공마다 무게 열 이름 (None = 모두 1, model_07부터 경기 안 공끼리 비슷한 만큼 줄여 셀 때)
    season_f: 새 시즌 첫 경기 전에 지난 시즌까지 누적 S·N에 곱하는 배수 (1 = 그대로, model_10부터 시즌 초)
      -> 조회할 때도 같은 값을 lookup(season_f=)에 넘겨야 함 (마지막 기록이 지난 시즌이면 거기서 곱함)
    cache: 캐시 키가 id(data)라 함수 안에서 매번 새로 만드는 임시 데이터는 cache=False (사라진 객체 id를 새 객체가 받으면 옛 표를 씀, model_11에서 발견)"""
    ck = (id(data), tuple(keys), tuple(cols), half_life, weight, season_f)
    if cache and ck in _TABLES:
        return _TABLES[ck]
    d = data.sort_values(["game_date", "gameID", "pit_no"])
    X = d[cols].to_numpy(float)
    ok = ~np.isnan(X)
    X = np.where(ok, X, 0.0)
    if weight is not None:
        wt = d[weight].to_numpy(float)[:, None]  # 공마다 무게
        X, ok = X * wt, ok * wt  # 합계·개수 모두 무게만큼
    dates = d["game_date"].to_numpy()
    seasons = d["game_date"].dt.year.to_numpy()  # 시즌 = 경기 날짜 연도
    decay = 0.5 ** (1 / half_life) if half_life else 1.0
    recs = []
    for key, idx in d.groupby(keys, sort=False).indices.items():
        key = key if isinstance(key, tuple) else (key,)
        S, N = np.zeros(len(cols)), np.zeros(len(cols))
        dd, ss = dates[idx], seasons[idx]
        starts = np.flatnonzero(np.r_[True, dd[1:] != dd[:-1]])
        for j, (s, e) in enumerate(zip(starts, np.r_[starts[1:], len(idx)])):
            if season_f != 1.0 and j > 0 and ss[s] != ss[starts[j - 1]]:
                S, N = S * season_f, N * season_f  # 새 시즌 첫 경기 -> 지난 시즌까지 누적 무게 줄임
            blk = idx[s:e]
            w = decay ** np.arange(len(blk) - 1, -1, -1)
            S = S * decay ** len(blk) + w @ X[blk]
            N = N * decay ** len(blk) + w @ ok[blk]
            recs.append((*key, dd[s], ss[s], *S, *N))
    t = pd.DataFrame(recs, columns=keys + ["game_date", "_season"] + ["S_" + c for c in cols] + ["N_" + c for c in cols])
    for k in keys + ["game_date"]:
        t[k] = t[k].astype(data[k].dtype)
    t = t.sort_values("game_date")
    if cache:
        _TABLES[ck] = t
    return t


def lookup(table, query, keys, cols, season_f=1.0):
    """각 행의 경기 날짜 전날까지 누적 S, N (같은 날 제외, 기록 없으면 0)
    season_f: prior_table과 같은 값, 마지막 기록이 지난 시즌이면 (= 이번 시즌 첫 경기) S·N에 곱함"""
    q = query[keys + ["game_date"]].copy()
    q["_row"] = np.arange(len(q))
    r = pd.merge_asof(q.sort_values("game_date"), table, on="game_date", by=keys,
                      allow_exact_matches=False).sort_values("_row")
    S, N = r[["S_" + c for c in cols]].fillna(0).to_numpy(), r[["N_" + c for c in cols]].fillna(0).to_numpy()
    if season_f != 1.0:
        f = np.where(r["_season"].notna() & (r["_season"] != r["game_date"].dt.year), season_f, 1.0)[:, None]
        S, N = S * f, N * f
    return S, N


def shrink(S, N, prior, k):
    return (S + k * prior) / (N + k)  # 표본이 적으면 상위 그룹 값 쪽으로


# ---------------------------------------------------------------- 확률 채점
def prob_metrics(P, y, ks=(1, 2, 3)):
    p_true = P[np.arange(len(P)), y]
    rank = (P > p_true[:, None]).sum(1)
    res = {"logloss": -np.log(np.clip(p_true, 1e-12, None)).mean()}
    res.update({f"top{k}": (rank < k).mean() for k in ks})
    return res


def zone_metrics(P, y):
    r = prob_metrics(P, y, ks=(1, 3, 5))
    g = prob_metrics(P @ GROUP2, pd.Series(ZONE_GROUP[y]).map({"H": 0, "S": 1, "C": 2}).to_numpy(), ks=(1,))
    r.update({"H/S/C logloss": g["logloss"], "H/S/C 정확도": g["top1"]})
    return r


def class_report(y, yhat, labels):
    """1등 예측 기준 precision·recall·F1, 전체 지표, 비율 혼동행렬 (행 = 실제)"""
    K = len(labels)
    cm = np.zeros((K, K), int)
    np.add.at(cm, (y, yhat), 1)
    tp, true_n, pred_n = np.diag(cm), cm.sum(1), cm.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rep = pd.DataFrame({"실제 개수": true_n, "예측 개수": pred_n, "precision": tp / pred_n,
                            "recall": tp / true_n, "F1": 2 * tp / (true_n + pred_n)}, index=labels)
    rep = rep[rep["실제 개수"] > 0]
    rep["F1"] = rep["F1"].fillna(0)
    total = {"accuracy": tp.sum() / len(y), "balanced accuracy": rep["recall"].mean(), "macro F1": rep["F1"].mean()}
    cmn = pd.DataFrame(cm / np.maximum(true_n[:, None], 1), index=labels, columns=labels).loc[rep.index, rep.index]
    return rep, total, cmn


# ---------------------------------------------------------------- 위치 채점
def gauss(M, q):
    """모멘트(x, zn, xx, zz, xz) → 2차원 정규분포 (중심 cm, 분산, 공분산)"""
    mx, mz, xx, zz, xz = M[:, :5].T
    h = (q["TOP_SZ_CM"] - q["BOTTOM_SZ_CM"]).to_numpy()
    vx = np.maximum(xx - mx ** 2, 4.0)
    vz = np.maximum(zz - mz ** 2, 1e-3) * h ** 2
    c = (xz - mx * mz) * h
    c = np.clip(c, -0.99 * np.sqrt(vx * vz), 0.99 * np.sqrt(vx * vz))
    return mx, q["BOTTOM_SZ_CM"].to_numpy() + mz * h, vx, vz, c


def coord_metrics(g, q):
    mx, mz, vx, vz, c = g
    dx, dz = q["x"].to_numpy() - mx, q["CROSS_PLATE_Z_CM"].to_numpy() - mz
    det = vx * vz - c ** 2
    nll = 0.5 * ((vz * dx ** 2 - 2 * c * dx * dz + vx * dz ** 2) / det + np.log(det)) + np.log(2 * np.pi)
    dist = np.hypot(dx, dz)
    return {"평균거리 cm": dist.mean(), "중앙거리 cm": np.median(dist),
            "RMSE x": np.sqrt((dx ** 2).mean()), "RMSE z": np.sqrt((dz ** 2).mean()), "NLL": nll.mean()}


def log_dens(g, q):
    """실제 공 위치의 로그 확률밀도 = −(coord_metrics의 NLL)"""
    mx, mz, vx, vz, c = g
    dx, dz = q["x"].to_numpy() - mx, q["CROSS_PLATE_Z_CM"].to_numpy() - mz
    det = vx * vz - c ** 2
    return -0.5 * ((vz * dx ** 2 - 2 * c * dx * dz + vx * dz ** 2) / det + np.log(det)) - np.log(2 * np.pi)


def zone_from_gauss(g, q):
    """정규분포 → 24칸 확률 (좌우·높이 독립 가정)"""
    mx, mz, vx, vz, _ = g
    ex = np.r_[-np.inf, X_EDGES, np.inf]
    Px = np.diff(ndtr((ex[None, :] - mx[:, None]) / np.sqrt(vx)[:, None]), axis=1)
    inf = np.full(len(q), np.inf)
    ez = np.column_stack([-inf, z_edges(q["TOP_SZ_CM"].to_numpy(), q["BOTTOM_SZ_CM"].to_numpy()), inf])
    Pz = np.diff(ndtr((ez - mz[:, None]) / np.sqrt(vz)[:, None]), axis=1)
    return (Pz[:, :, None] * Px[:, None, :]).reshape(len(q), 36) @ CELL2ZONE


def zone_from_gauss_cov(g, q, m=32):
    """정규분포 → 24칸 확률 (좌우·높이 공분산 반영, model_03부터)
    P(칸) = ∫ 좌우 구간 P(x) · P(높이 구간 | x) dx, 좌우 누적확률 위에서 가우스-르장드르 m점"""
    mx, mz, vx, vz, c = g  # 중심 (좌우, 높이 cm), 분산, 공분산
    sx, sz = np.sqrt(vx), np.sqrt(vz)  # 표준편차
    rho = c / (sx * sz)  # 좌우·높이 상관계수 (gauss에서 ±0.99로 잘려 있음)
    sd_c = sz * np.sqrt(1 - rho ** 2)  # 좌우를 알 때 높이 표준편차
    ux = ndtr((np.r_[-np.inf, X_EDGES, np.inf][None, :] - mx[:, None]) / sx[:, None])  # 좌우 경계 7개의 누적확률 (n, 7)
    t, w = np.polynomial.legendre.leggauss(m)  # [-1, 1] 구간 가우스-르장드르 점·가중치 (m개)
    u = ux[:, :-1, None] + (ux[:, 1:, None] - ux[:, :-1, None]) * (t + 1) / 2  # 좌우 구간 6개마다 누적확률 위의 적분 점 (n, 6, m)
    du = (ux[:, 1:] - ux[:, :-1])[:, :, None] * w / 2  # 각 점이 맡는 확률 (n, 6, m), 구간별 합 = 그 좌우 구간 확률
    e = ndtri(np.clip(u, 1e-15, 1 - 1e-15))  # 그 점의 좌우 표준화 값 (누적확률 -> 표준정규 값)
    mu_c = mz[:, None, None] + (rho * sz)[:, None, None] * e  # 좌우를 알 때 높이 중심 (n, 6, m)
    inf = np.full(len(q), np.inf)
    ez = np.column_stack([-inf, z_edges(q["TOP_SZ_CM"].to_numpy(), q["BOTTOM_SZ_CM"].to_numpy()), inf])  # 높이 경계 (n, 7), 그 타자 존 기준
    Fz = ndtr((ez[:, None, None, :] - mu_c[..., None]) / sd_c[:, None, None, None])  # 높이 경계의 조건부 누적확률 (n, 6, m, 7)
    Pz = np.diff(Fz, axis=3)  # 높이 구간 6개의 조건부 확률 (n, 6, m, 6)
    P36 = np.einsum("nxk,nxkz->nzx", du, Pz).reshape(len(q), 36)  # 적분 점 합 → 36칸 (zi*6 + xi, CELL_NAMES 순서)
    return P36 @ CELL2ZONE  # 36칸 → 24칸 (바깥 칸 합치기)


def as_type(q, t):
    """구종만 t로 바꾼 사본 (t: 구종 이름 하나 또는 공마다 이름 배열)"""
    qt = q.copy()
    qt["pitch_type"] = pd.Series(np.broadcast_to(t, len(q)), index=q.index).astype(q["pitch_type"].dtype)
    return qt


# ---------------------------------------------------------------- 보정
def calib(p, hit):
    """10% 구간별 예측 평균 vs 실제 비율, ECE"""
    b = np.clip(np.digitize(p, BINS) - 1, 0, len(BINS) - 2)
    t = (pd.DataFrame({"b": b, "p": p, "hit": hit}).groupby("b")
         .agg(공수=("p", "size"), 예측평균=("p", "mean"), 실제비율=("hit", "mean")))
    t.index = [f"{BINS[i]:.0%}~{BINS[i + 1]:.0%}" for i in t.index]
    t["차이"] = t["실제비율"] - t["예측평균"]
    return t, (t["공수"] * t["차이"].abs()).sum() / t["공수"].sum()


def calib_report(P, y, labels):
    tables, rows = {}, []
    for k, lab in enumerate(labels):
        tables[lab], e = calib(P[:, k], y == k)
        rows.append({"항목": lab, "실제 비율": (y == k).mean(), "예측 평균": P[:, k].mean(), "ECE": e})
    hit = P.argmax(1) == y
    tables["1등 예측"], e = calib(P.max(1), hit)
    rows.append({"항목": "1등 예측", "실제 비율": hit.mean(), "예측 평균": P.max(1).mean(), "ECE": e})
    return pd.DataFrame(rows).set_index("항목"), tables


def calib_plot(tables, title, ax, show=None, min_n=30):
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    for lab, t in tables.items():
        if show is not None and lab not in show:
            continue
        t = t[t["공수"] >= min_n]
        ax.plot(t["예측평균"], t["실제비율"], marker="o", ms=3, lw=2.5 if lab == "1등 예측" else 1,
                color="black" if lab == "1등 예측" else None, label="top1" if lab == "1등 예측" else lab)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="predicted", ylabel="observed", title=title)
    ax.legend(fontsize=7)


def brier_decomp(P, y, labels, bins=BINS):
    """항목별 Brier = 보정 오차 − 구분력 + 불확실성 (구간 기준 근사)"""
    rows = []
    for k, lab in enumerate(labels):
        p, o = P[:, k], (y == k).astype(float)
        b = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
        g = pd.DataFrame({"b": b, "p": p, "o": o}).groupby("b").agg(n=("p", "size"), pm=("p", "mean"), om=("o", "mean"))
        N, obar = len(p), o.mean()
        rows.append({"항목": lab, "Brier": np.mean((p - o) ** 2),
                     "보정 오차": (g.n * (g.pm - g.om) ** 2).sum() / N,
                     "구분력": (g.n * (g.om - obar) ** 2).sum() / N,
                     "불확실성": obar * (1 - obar)})
    t = pd.DataFrame(rows).set_index("항목")
    t.loc["합계"] = t.sum()
    return t


def fit_recal(P, y, iters=50):
    """재보정 softmax(a × log P + b)를 Newton법으로 맞춘 함수 반환"""
    L = np.log(np.clip(P, 1e-12, None))
    n, K = P.shape
    Y = np.eye(K)[y]
    G = np.concatenate([L[:, :, None], np.broadcast_to(np.eye(K), (n, K, K))], 2)
    loss = lambda th: -log_softmax(th[0] * L + th[1:], axis=1)[np.arange(n), y].mean()
    th = np.r_[1.0, np.zeros(K)]
    for _ in range(iters):
        S = np.exp(log_softmax(th[0] * L + th[1:], axis=1))
        grad = np.einsum("nk,nkp->p", S - Y, G) / n
        W = S[:, :, None] * np.eye(K) - S[:, :, None] * S[:, None, :]
        H = np.einsum("nkp,nkl,nlq->pq", G, W, G) / n + 1e-6 * np.eye(K + 1)
        step, f0 = np.linalg.solve(H, grad), loss(th)
        t = 1.0
        while loss(th - t * step) > f0 and t > 1e-4:
            t /= 2
        th = th - t * step
        if np.abs(t * step).max() < 1e-8:
            break
    return lambda Pn: np.exp(log_softmax(th[0] * np.log(np.clip(Pn, 1e-12, None)) + th[1:], axis=1))


def recal_report(Pv, yv, Pt_, yt_):
    """검증 연도로 재보정을 맞추고 평가 연도에 적용"""
    f = fit_recal(Pv, yv)
    ll = lambda P, y: -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).mean()
    raw, rec = ll(Pt_, yt_), ll(f(Pt_), yt_)
    return {"log loss (원래)": raw, "log loss (재보정 후)": rec, "보정으로 줄일 수 있던 몫": raw - rec}


# ---------------------------------------------------------------- 실전 출력
def show_types(P, thr=0.8, k=None):
    """공마다 보여 줄 구종 (확률 높은 순으로 누적 thr까지, k를 주면 상위 k개 고정)"""
    order = np.argsort(-P, 1)
    if k is None:
        cum = np.take_along_axis(P, order, 1).cumsum(1)
        k = (cum < thr).sum(1) + 1
    show = np.zeros(P.shape, bool)
    np.put_along_axis(show, order, np.arange(P.shape[1])[None, :] < np.reshape(k, (-1, 1)), 1)
    return show


# ---------------------------------------------------------------- 2단계 변수 (구종 모델 입력)
# 정보 규칙 (README 3장 "평가 설계"): 경기 기록은 그 공 앞까지, 트래킹은 전날까지 (같은 날 더블헤더 1차전 트래킹도 X)
# 경기 기록 열: 사람이 현장에서 바로 아는 값 (나머지 열은 전부 트래킹으로 봄)
GAME_COLS = ["gameID", "GYEAR", "game_date", "pit_no", "pa_id", "inning", "tb", "bat_team", "batorder", "bat_turn",
             "pitcher_code", "batter_code", "catcher_code", "p_throws", "STANCE", "same_side", "strike_before", "ball_before",
             "out_cn", "base1", "base2", "base3", "away_score", "home_score", "VENUE_ID"]
RESULT_COLS = ["pit_result", "how", "field_if", "place_sc"]  # 경기 기록이지만 그 공의 결과 (그 공 자신에게는 미래)
A_COLS = ["pc", "tto", "starter", "inning", "out_cn", "base1", "base2", "base3", "batorder", "sdiff"]  # A 이번 타석 상황


def situation_features(m, q):
    """A 이번 타석 상황 (model_09부터): 경기 기록만, 그 공 앞 정보
    m: 모든 공 (구종 의심 공 포함, 투구 수 세는 데 씀), q: 변수를 붙일 공 (KEYS 포함, 보통 초구) -> q 순서의 (len(q), A_COLS)
      pc       그 경기 그 투수의 앞선 투구 수 (0 = 그 경기 첫 타자)
      tto      같은 타자 몇 번째 대결 (그 경기, 이 타석 포함)
      starter  그 경기 이 투수의 첫 공이 1회 = 선발
      sdiff    투수 팀 기준 점수 차 (투구 전)
      나머지   그 공 행의 투구 전 상태 (이닝, 아웃, 주자, 타순)"""
    mm = m[GAME_COLS].sort_values(KEYS)
    g = mm.groupby(["gameID", "pitcher_code"], sort=False)
    gb = mm.groupby(["gameID", "pitcher_code", "batter_code"], sort=False)
    new_pa = mm["pa_id"].ne(gb["pa_id"].shift()).astype(int)  # 같은 타자와 새 타석 시작 (자기 앞 행과 비교)
    ctx = pd.DataFrame({
        "pc": g.cumcount(),  # 앞선 공 수 (자기 앞 행만 셈)
        "starter": g["inning"].transform("first").eq(1).astype(int),  # 그 경기 첫 공 (자기 이전 행)
        "tto": new_pa.groupby([mm["gameID"], mm["pitcher_code"], mm["batter_code"]], sort=False).cumsum(),  # 같은 타자 몇 번째 대결
        "sdiff": np.where(mm["tb"].eq("T"), mm["home_score"] - mm["away_score"], mm["away_score"] - mm["home_score"]),
    }, index=mm.index)
    keep = ["inning", "out_cn", "base1", "base2", "base3", "batorder"]
    X = q[KEYS].merge(pd.concat([mm[KEYS + keep], ctx], axis=1), on=KEYS, how="left", validate="one_to_one")
    X.index = q.index
    return X[A_COLS]


HIST_MIX_HL = 200.0  # 모든 카운트 배합 반감기 (공 수)


def pitcher_history_features(m, q, mix_hl=HIST_MIX_HL):
    """C 투수 이력 (model_11부터): 경기 기록은 그 경기 전, 트래킹(구종)은 전날까지 -> q 순서의 DataFrame
      rest          직전 등판(이전 날짜)과의 간격 (일, 첫 등판은 빈 값)
      role_change   직전 등판과 선발/불펜이 다름 (1/0, 첫 등판은 빈 값), 선발 = 그 경기 첫 공이 1회 (경기 전 발표)
      mix_all_<구종>  모든 카운트 구종 배합 (전날까지, 반감 mix_hl구, 구종 의심 공 제외)
      mix_last_<구종> 직전 날짜 경기의 모든 카운트 구종 배합 (그 경기 공이 없으면 빈 값)"""
    mm = m.sort_values(KEYS)
    gd = (mm.groupby(["pitcher_code", "gameID"], sort=False)
          .agg(game_date=("game_date", "first"), starter=("inning", "first")).reset_index().sort_values("game_date"))
    gd["starter"] = gd["starter"].eq(1).astype(int)  # 그 경기 첫 공이 1회
    prev = gd[["pitcher_code", "game_date", "starter"]].rename(columns={"starter": "prev_starter"}).assign(prev_date=gd["game_date"])
    gd = pd.merge_asof(gd, prev, on="game_date", by="pitcher_code", allow_exact_matches=False)  # 이전 날짜 등판 (같은 날 1차전 X)
    gd["rest"] = (gd["game_date"] - gd["prev_date"]).dt.days
    gd["role_change"] = np.where(gd["prev_starter"].isna(), np.nan, (gd["starter"] != gd["prev_starter"]).astype(float))
    X = q[["pitcher_code", "gameID"]].merge(gd[["pitcher_code", "gameID", "rest", "role_change"]], on=["pitcher_code", "gameID"],
                                            how="left", validate="many_to_one")
    X.index = q.index
    ok = mm[mm["label_doubt"].eq(False)].copy()  # 구종 의심 공 제외 (빈 값 = 오늘 트래킹 -> 제외)
    ok[CLASSES] = (ok["pitch_type"].to_numpy()[:, None] == np.array(CLASSES)[None, :]).astype(float)
    S, N = lookup(prior_table(ok, ["pitcher_code"], CLASSES, mix_hl, cache=False), q, ["pitcher_code"], CLASSES)  # ok는 매번 새 객체 -> 캐시 X
    with np.errstate(invalid="ignore"):
        X[["mix_all_" + c for c in CLASSES]] = np.where(N > 0, S / np.where(N > 0, N, 1), np.nan)
    lg = (ok.groupby(["pitcher_code", "gameID"]).agg(game_date=("game_date", "first"), **{c: (c, "mean") for c in CLASSES})
          .reset_index().sort_values("game_date"))
    qq = q[["pitcher_code", "game_date"]].assign(_r=np.arange(len(q))).sort_values("game_date")
    last = pd.merge_asof(qq, lg.drop(columns="gameID"), on="game_date", by="pitcher_code", allow_exact_matches=False).sort_values("_r")
    X[["mix_last_" + c for c in CLASSES]] = last[CLASSES].to_numpy()  # 직전 날짜 경기 (전날까지)
    return X.drop(columns=["pitcher_code", "gameID"])


SWING = ["S", "F", "H", "W", "V"]  # 스윙한 결과 (헛스윙·파울·타격·번트 파울·번트 헛스윙)


def opponent_features(m, q, swing_k=20.0):
    """D 상대 (model_11부터, 경기 기록만): q 순서의 DataFrame
      home       이 투수 팀이 홈 (초 수비)
      bat_swing  그 타자의 전날까지 0-0 공 스윙률 (41명 투수 공만, 그 공 전날까지 리그 0-0 스윙률 쪽으로 swing_k구 당김)
      bat_n      그 타자의 전날까지 0-0 공 수"""
    X = pd.DataFrame({"home": q["tb"].eq("T").astype(int)}, index=q.index)
    m00 = m.loc[m["ball_before"].eq(0) & m["strike_before"].eq(0), ["batter_code", "game_date", "gameID", "pit_no"]].copy()
    m00["swing"] = m.loc[m00.index, "pit_result"].isin(SWING).astype(float)  # 결과 열이 빈 값(그 공 자신)이면 False지만 같은 날이라 안 셈
    S, N = lookup(prior_table(m00, ["batter_code"], ["swing"], cache=False), q, ["batter_code"], ["swing"])  # m00은 매번 새 객체
    m00["_all"] = 1
    Sa, Na = lookup(prior_table(m00, ["_all"], ["swing"], cache=False), q.assign(_all=1), ["_all"], ["swing"])
    lg = np.where(Na[:, 0] > 0, Sa[:, 0] / np.maximum(Na[:, 0], 1), 0.35)  # 리그 0-0 스윙률: 공마다 그 공의 전날까지 (기록 없으면 0.35)
    X["bat_swing"] = (S[:, 0] + swing_k * lg) / (N[:, 0] + swing_k)
    X["bat_n"] = N[:, 0]
    return X


def group_residual(data, P, keys, q, k=50.0, prefix="enc"):
    """그룹(타자·상대 팀 등)별 전날까지 (실제 원-핫 − P) 합 / (공 수 + k) -> q 순서의 (len(q), 9) DataFrame (model_11부터)
    = 그 그룹을 만났을 때 지금 확률 P가 구종별로 빗나간 정도 (기록 적으면 0 쪽으로)
    data: P와 같은 순서의 초구 (fp 일부), P: (len(data), 9) 그 공의 전날까지 정보로 만든 확률 (예: B4), 빈 값 행은 안 셈"""
    R = data[KEYS + keys + ["game_date"]].copy()
    R[["r_" + c for c in CLASSES]] = data[CLASSES].to_numpy() - P  # 잔차 (P가 빈 값이면 빈 값 -> prior_table이 안 셈)
    cols = ["r_" + c for c in CLASSES]
    S, N = lookup(prior_table(R, keys, cols, cache=False), q, keys, cols)  # R은 매번 새 객체 -> 캐시 X
    return pd.DataFrame(S / (N[:, :1] + k), index=q.index, columns=[f"{prefix}_{c}" for c in CLASSES])


FIRST_RES = {"B": "볼", "T": "루킹", "S": "헛스윙", "V": "헛스윙", "F": "파울", "W": "파울", "H": "인플레이"}  # 앞 타석 초구 결과 묶음
HOW_GRP = {**{h: "안타" for h in ["H1", "H2", "H3", "HI", "HB"]}, "HR": "홈런", **{h: "볼넷·사구" for h in ["BB", "IB", "HP"]},
           **{h: "삼진" for h in ["KK", "KN", "KW", "KB", "KP"]}}  # 앞 타석 결과 묶음 (나머지 = 아웃·기타)
FIRST_CATS = ["없음", "볼", "루킹", "헛스윙", "파울", "인플레이"]
HOW_CATS = ["없음", "안타", "홈런", "볼넷·사구", "삼진", "아웃·기타"]


def game_flow_features(m, q):
    """B 오늘 경기 흐름 (model_11부터, 경기 기록만): 같은 경기·같은 투수의 그 타석 앞 정보 -> q 순서의 DataFrame
      ball_rate        그 경기 앞선 공 중 볼 비율 (앞 공 없으면 빈 값)
      fs_rate          그 경기 앞 타석들의 초구 스트라이크(볼 아님) 비율 (앞 타석 없으면 빈 값)
      prev_first_<결과> 바로 앞 타석 초구 결과 원-핫 (없음 = 그 경기 첫 타자)
      prev_how_<결과>   바로 앞 타석 결과 원-핫 (없음 = 그 경기 첫 타자)"""
    mm = m[GAME_COLS + RESULT_COLS].sort_values(KEYS)
    g = mm.groupby(["gameID", "pitcher_code"], sort=False)
    isB = mm["pit_result"].eq("B").astype(float)
    pc = g.cumcount()
    ball_rate = (isB.groupby([mm["gameID"], mm["pitcher_code"]], sort=False).cumsum() - isB) / pc.replace(0, np.nan)  # 자기 공 빼고 앞 공만
    pa = (mm.groupby(["gameID", "pitcher_code", "pa_id"], sort=False)
          .agg(first_res=("pit_result", "first"), how=("how", "last"), p0=("pit_no", "min")).reset_index().sort_values(["gameID", "p0"]))
    pa["first_g"] = pa["first_res"].map(FIRST_RES)
    pa["how_g"] = pa["how"].map(HOW_GRP).fillna("아웃·기타")
    gp = pa.groupby(["gameID", "pitcher_code"], sort=False)
    pa["prev_first"] = gp["first_g"].shift().fillna("없음")  # 바로 앞 타석 (자기 타석 X)
    pa["prev_how"] = gp["how_g"].shift().fillna("없음")
    strike = pa["first_g"].ne("볼").astype(float)
    pa["fs_rate"] = (strike.groupby([pa["gameID"], pa["pitcher_code"]], sort=False).cumsum() - strike) / gp.cumcount().replace(0, np.nan)  # 앞 타석만
    X = (q[KEYS + ["pitcher_code"]]
         .merge(mm[KEYS + ["pa_id"]].assign(ball_rate=ball_rate.to_numpy()), on=KEYS, how="left", validate="one_to_one")
         .merge(pa[["gameID", "pitcher_code", "pa_id", "prev_first", "prev_how", "fs_rate"]], on=["gameID", "pitcher_code", "pa_id"], how="left"))
    X.index = q.index
    out = X[["ball_rate", "fs_rate"]].copy()
    for c in FIRST_CATS:
        out["prev_first_" + c] = X["prev_first"].eq(c).astype(int)
    for c in HOW_CATS:
        out["prev_how_" + c] = X["prev_how"].eq(c).astype(int)
    return out


RECENT_COLS = {"strike": "스트라이크 비율", "whiff": "헛스윙 비율", "velo": "구속", "spin": "회전수"}  # 구종별 최근 효과 항목


def pitch_type_recent_features(m, q, hl_recent=30.0, hl_long=300.0):
    """구종별 최근 효과 (model_12부터, 트래킹 전날까지): 투수×구종 최근(반감 hl_recent구) − 평소(반감 hl_long구) -> q 순서의 DataFrame
      rec_<항목>_<구종>: 항목 = strike(볼 아님 비율)·whiff(헛스윙 비율)·velo(초속 km/h)·spin(회전수), 모든 카운트·구종 의심 공 제외
      그 구종 기록이 없으면 빈 값 -> 결과(strike·whiff)는 경기 기록이지만 구종이 트래킹이라 전날까지"""
    ok = m[m["label_doubt"].eq(False)].copy()  # 구종 의심 공 제외 (빈 값 = 오늘 트래킹 -> 제외)
    ok["pt"] = ok["pitch_type"].astype(str)
    ok["strike"] = (~ok["pit_result"].eq("B")).astype(float)
    ok["whiff"] = ok["pit_result"].isin(["S", "V"]).astype(float)
    ok["velo"], ok["spin"] = ok["PITCH_START_SPEED_KOR"].astype(float), ok["SPIN_RATE"].astype(float)
    cols = list(RECENT_COLS)
    qq = pd.concat([q[["pitcher_code", "game_date"]].assign(pt=t, _r=np.arange(len(q))) for t in CLASSES], ignore_index=True)  # 공마다 구종 9개
    mean = {}
    for nm, hl in (("recent", hl_recent), ("long", hl_long)):
        S, N = lookup(prior_table(ok, ["pitcher_code", "pt"], cols, hl, cache=False), qq, ["pitcher_code", "pt"], cols)  # ok는 매번 새 객체
        with np.errstate(invalid="ignore", divide="ignore"):
            mean[nm] = np.where(N > 0, S / N, np.nan)
    d = pd.DataFrame(mean["recent"] - mean["long"], columns=cols)
    d["pt"], d["_r"] = qq["pt"].to_numpy(), qq["_r"].to_numpy()
    w = d.pivot(index="_r", columns="pt", values=cols)  # (len(q), 항목 × 구종)
    w.columns = [f"rec_{a}_{b}" for a, b in w.columns]
    w = w.reindex(columns=[f"rec_{a}_{b}" for a in cols for b in CLASSES])
    w.index = q.index
    return w


# ---------------------------------------------------------------- 3단계 변수 (위치 보정 입력, model_18부터)
def batter_location_features(m, q, k=50.0):
    """타자 성향 (위치용, model_18부터): 그 타자의 전날까지 모든 카운트 공 (41명 투수 공만) -> q 순서의 DataFrame
      bat_chase   존 밖(C) 공 스윙률 / bat_zswing  존 안(H·S) 공 스윙률 -> 그 공 전날까지 리그 값 쪽으로 k구 당김
      bat_xbh     타석당 장타(2루타·3루타·홈런) 비율 -> 리그 쪽으로 k타석 당김
      위치는 트래킹이라 전날까지, 결과도 전날까지로 셈 (리그 기록이 없으면 빈 값)"""
    d = m[["batter_code", "game_date", "gameID", "pit_no", "zone_group", "pit_result"]].copy()
    d["swing"] = d["pit_result"].isin(SWING).astype(float)
    d["_all"] = 1
    qa = q.assign(_all=1)
    out = pd.DataFrame(index=q.index)

    def shrunk(data, col, name):  # 타자별 전날까지 비율을 리그 비율 쪽으로 k만큼
        S, N = lookup(prior_table(data, ["batter_code"], [col], cache=False), q, ["batter_code"], [col])  # data는 매번 새 객체 -> 캐시 X
        Sa, Na = lookup(prior_table(data, ["_all"], [col], cache=False), qa, ["_all"], [col])
        lg = np.where(Na[:, 0] > 0, Sa[:, 0] / np.maximum(Na[:, 0], 1), np.nan)  # 리그 비율: 공마다 그 공의 전날까지
        out[name] = (S[:, 0] + k * lg) / (N[:, 0] + k)

    shrunk(d[d["zone_group"].eq("C")], "swing", "bat_chase")
    shrunk(d[d["zone_group"].isin(["H", "S"])], "swing", "bat_zswing")
    pa = (m.sort_values(KEYS).groupby(["gameID", "pa_id"], sort=False)
          .agg(batter_code=("batter_code", "first"), game_date=("game_date", "first"), pit_no=("pit_no", "max"), how=("how", "last"))
          .reset_index())
    pa["xbh"] = pa["how"].isin(["H2", "H3", "HR"]).astype(float)
    pa["_all"] = 1
    shrunk(pa, "xbh", "bat_xbh")
    return out


SB_GRP = {**{h: "안타" for h in ["H1", "HI", "HB"]}, **{h: "장타" for h in ["H2", "H3", "HR"]},
          **{h: "볼넷·사구" for h in ["BB", "IB", "HP"]}, **{h: "삼진" for h in ["KK", "KN", "KW", "KB", "KP"]}}  # 이 타자와 앞 대결 결과 묶음
SB_CATS = ["없음", "안타", "장타", "볼넷·사구", "삼진", "아웃·기타"]


def same_batter_features(m, q):
    """이 타자와 그 경기 앞 대결 (위치용, model_18부터, 경기 기록만): 같은 경기·같은 투수·같은 타자의 바로 앞 타석 -> q 순서의 DataFrame
      sb_prev_<결과>: 바로 앞 대결 결과 원-핫 (없음 = 오늘 첫 대결, 장타 = 2루타·3루타·홈런)"""
    mm = m[GAME_COLS + RESULT_COLS].sort_values(KEYS)
    pa = (mm.groupby(["gameID", "pitcher_code", "batter_code", "pa_id"], sort=False)
          .agg(how=("how", "last"), p0=("pit_no", "min")).reset_index().sort_values(["gameID", "p0"]))
    pa["res"] = pa["how"].map(SB_GRP).fillna("아웃·기타")
    pa["sb_prev"] = pa.groupby(["gameID", "pitcher_code", "batter_code"], sort=False)["res"].shift().fillna("없음")  # 바로 앞 대결 (자기 타석 X)
    X = (q[KEYS].merge(mm[KEYS + ["pa_id", "pitcher_code", "batter_code"]], on=KEYS, how="left", validate="one_to_one")
         .merge(pa[["gameID", "pitcher_code", "batter_code", "pa_id", "sb_prev"]], on=["gameID", "pitcher_code", "batter_code", "pa_id"], how="left"))
    X.index = q.index
    return pd.DataFrame({"sb_prev_" + c: X["sb_prev"].eq(c).astype(int) for c in SB_CATS}, index=q.index)


def appearance_features(m, q):
    """그 시즌 몇 번째 등판 (위치용, model_18부터, 경기 기록): 41명 투수 데이터에 있는 경기 기준, 그 경기 포함 -> app_no"""
    gd = (m.groupby(["pitcher_code", "gameID"], sort=False).agg(game_date=("game_date", "first")).reset_index()
          .sort_values(["game_date", "gameID"]))
    gd["app_no"] = gd.groupby(["pitcher_code", gd["game_date"].dt.year]).cumcount() + 1
    X = q[["pitcher_code", "gameID"]].merge(gd[["pitcher_code", "gameID", "app_no"]], on=["pitcher_code", "gameID"], how="left", validate="many_to_one")
    X.index = q.index
    return X[["app_no"]]


def group_residual_loc(data, P3, keys, q, k=50.0, half_life=None, prefix="loc"):
    """그룹별 전날까지 H/S/C (실제 원-핫 − P3) 합 / (공 수 + k) -> q 순서의 (len(q), 3) DataFrame (위치용, model_18부터)
    = 그 그룹(포수·구장·투수×상황·리그 전체 등)에서 지금 위치 확률이 H/S/C별로 빗나간 정도 (half_life 주면 최근 가중)
    data: P3와 같은 순서의 초구 (zone_group·keys 열 포함), P3: (len(data), 3) 그 공의 전날까지 정보로 만든 H/S/C 확률, 빈 값 행은 안 셈"""
    R = data[KEYS + keys + ["game_date"]].copy()
    cols = ["r_" + c for c in HSC]
    R[cols] = (data["zone_group"].to_numpy()[:, None] == np.array(HSC)[None, :]).astype(float) - P3  # 잔차 (P3가 빈 값이면 안 셈)
    S, N = lookup(prior_table(R, keys, cols, half_life, cache=False), q, keys, cols)  # R은 매번 새 객체 -> 캐시 X
    return pd.DataFrame(S / (N[:, :1] + k), index=q.index, columns=[f"{prefix}_{c}" for c in HSC])


def as_of(m, row):
    """그 공을 던지기 직전에 알 수 있는 데이터만 남긴 m 사본 (미래 정보 검사용)
    - 전날까지 경기: 전부
    - 같은 날 다른 경기(더블헤더)·같은 경기 앞 공: 경기 기록 + 결과 열만 (트래킹은 비움)
    - 그 공 자신: 경기 기록만 (결과·트래킹 비움)
    - 그 뒤 공: 없음"""
    d, gid, no = row["game_date"], row["gameID"], row["pit_no"]
    past = m["game_date"] < d
    today = m["game_date"].eq(d) & (m["gameID"].ne(gid) | (m["pit_no"] < no))
    me = m["gameID"].eq(gid) & m["pit_no"].eq(no)
    t = m[past | today | me].copy()
    track = [c for c in t.columns if c not in GAME_COLS + RESULT_COLS]
    def blank(rows, cols):  # 그 행·열을 빈 값으로 (실수 열은 그대로, 나머지는 object로 바꿔서)
        other = [c for c in cols if not pd.api.types.is_float_dtype(t[c])]
        t[other] = t[other].astype(object)
        t.loc[rows, cols] = np.nan

    blank((t["game_date"] == d).to_numpy(), track)  # 오늘 트래킹 전부 비움
    blank((t["gameID"].eq(gid) & t["pit_no"].eq(no)).to_numpy(), RESULT_COLS)  # 그 공의 결과 비움
    return t


def check_no_future(feature_fn, m, q, n=100, seed=0, **kw):
    """미래 정보 검사: 공 n개를 뽑아 그 공 직전까지 자른 데이터(as_of)로 변수를 다시 계산 -> 전체 데이터로 계산한 값과 같아야 함
    feature_fn(m, q, **kw) -> q 순서의 DataFrame. 다른 공이 있으면 그 공과 열 목록을 돌려줌 (빈 표 = 통과)"""
    full = feature_fn(m, q, **kw)
    rng = np.random.default_rng(seed)
    first = q.groupby(["gameID", "pitcher_code"]).head(1).index  # 그 경기 첫 타자 (앞 공 없음) 일부 포함
    pick = np.r_[rng.choice(q.index, n - n // 5, replace=False), rng.choice(first, n // 5, replace=False)]
    bad = []
    for i in pick:
        row = q.loc[i]
        t = as_of(m, row)
        qi = q.loc[[i]]
        got = feature_fn(t, qi, **kw).iloc[0]
        exp = full.loc[i]
        diff = [c for c in full.columns if not (pd.isna(exp[c]) and pd.isna(got[c])) and not np.isclose(float(exp[c]), float(got[c]))]
        if diff:
            bad.append({"gameID": row["gameID"], "pit_no": row["pit_no"], "다른 열": diff})
    print(f"미래 정보 검사: {len(pick)}개 중 다른 공 {len(bad)}개")
    return pd.DataFrame(bad)


# ---------------------------------------------------------------- 결과 파일
def save_results(model, per_pitch, tables=None):
    """results/<model>/per_pitch.parquet와 요약 표(csv) 저장"""
    miss = [c for c in KEYS + P_TYPE + P_ZONE if c not in per_pitch]
    assert not miss, f"필수 열 없음: {miss[:5]}"
    assert not per_pitch.duplicated(KEYS).any(), "같은 공이 두 번 들어 있음"
    assert np.allclose(per_pitch[P_TYPE].sum(1), 1, atol=1e-6) and np.allclose(per_pitch[P_ZONE].sum(1), 1, atol=1e-6), "확률 합이 1이 아님"
    d = os.path.join(RESULTS_DIR, model)
    os.makedirs(d, exist_ok=True)
    per_pitch.to_parquet(os.path.join(d, "per_pitch.parquet"), index=False)
    for name, t in (tables or {}).items():
        t.to_csv(os.path.join(d, name + ".csv"), encoding="utf-8-sig")
    print("저장:", d, sorted(os.listdir(d)))


def load_results(models=None):
    """results/*/per_pitch.parquet 전부 (베이스라인 먼저)"""
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "*", "per_pitch.parquet")))
    out = {os.path.basename(os.path.dirname(p)): p for p in paths}
    names = sorted(out, key=lambda s: (s != "baseline", s))
    return {n: pd.read_parquet(out[n]) for n in names if models is None or n in models}


def align(pp, fp, year=TEST_YEAR):
    """그 해 초구 전부(fp 순서)에 공별 예측 붙이기 -> 실제값과 예측이 한 행에"""
    q = fp[fp["GYEAR"].eq(year)].reset_index(drop=True)
    pred = pp[[c for c in pp if c in KEYS or c not in q.columns]]
    d = q.merge(pred, on=KEYS, how="left", validate="one_to_one")
    assert d[P_TYPE[0]].notna().all(), f"{year} 초구 중 예측 없는 공 {d[P_TYPE[0]].isna().sum()}개"
    return d


def evaluate(pp, fp, year=TEST_YEAR):
    """한 모델의 주요 지표 (없는 선택 열은 건너뜀)"""
    d = align(pp, fp, year)
    yt, yz = d["pitch_type"].map(TYPE_IDX).to_numpy(), d["zone"].map(ZONE_IDX).to_numpy()
    Pt, Pz = d[P_TYPE].to_numpy(), d[P_ZONE].to_numpy()
    t, z = prob_metrics(Pt, yt), zone_metrics(Pz, yz)
    r = {"구종 logloss": t["logloss"], "구종 top1": t["top1"], "구종 top2": t["top2"], "구종 top3": t["top3"],
         "구종 macro F1": class_report(yt, Pt.argmax(1), CLASSES)[1]["macro F1"],
         "24칸 logloss (실전)": z["logloss"], "24칸 top1": z["top1"], "24칸 top3": z["top3"],
         "H/S/C logloss": z["H/S/C logloss"], "H/S/C 정확도": z["H/S/C 정확도"]}
    if "ll_zone_true" in d:
        r["24칸 logloss (실제 구종)"] = d["ll_zone_true"].mean()
    if "ll_joint" in d:
        r["구종×칸 logloss"] = d["ll_joint"].mean()
    if "dist" in d:
        r["좌표 평균거리 cm"] = d["dist"].mean()
    if "nll_coord" in d:
        r["좌표 NLL"] = d["nll_coord"].mean()
    return r


SCORE_COLS = ["ll_type", "ll_zone", "ll_hsc", "ll_joint", "nll_coord", "dist"]  # 공별 점수 (작을수록 좋음)
# ll_hsc (2026-11-11 추가): 24칸 확률을 H/S/C로 합친 실전 log loss -> 채택 조건 (다): 24칸으로 고르고 H/S/C가 나빠지지 않아야 함


def pitch_scores(pp, fp, year=TEST_YEAR):
    """공별 점수 (부트스트랩용): 구종·24칸·H/S/C log loss는 확률에서 계산, 나머지는 있으면 그대로"""
    d = align(pp, fp, year)
    i = np.arange(len(d))
    nll = lambda p: -np.log(np.clip(p, 1e-12, None))
    yz = d["zone"].map(ZONE_IDX).to_numpy()
    s = pd.DataFrame({"gameID": d["gameID"],
                      "ll_type": nll(d[P_TYPE].to_numpy()[i, d["pitch_type"].map(TYPE_IDX).to_numpy()]),
                      "ll_zone": nll(d[P_ZONE].to_numpy()[i, yz]),
                      "ll_hsc": nll((d[P_ZONE].to_numpy() @ GROUP2)[i, pd.Series(ZONE_GROUP[yz]).map({"H": 0, "S": 1, "C": 2}).to_numpy()])})
    for c in SCORE_COLS[3:]:
        if c in d:
            s[c] = d[c].to_numpy()
    return s


def bootstrap_diff(sa, sb, B=1000, seed=0):
    """경기 단위 부트스트랩: 모델 b − 모델 a 공별 점수 평균 차이와 95% 구간 (음수 = b가 좋음)"""
    cols = [c for c in SCORE_COLS if c in sa and c in sb]
    ga, gb = sa.groupby("gameID")[cols].sum(), sb.groupby("gameID")[cols].sum()
    n = sa.groupby("gameID").size()
    D, n = (gb - ga).loc[n.index].to_numpy(), n.to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(n), (B, len(n)))  # 경기를 중복 허용해 다시 뽑기
    boot = D[idx].sum(1) / n[idx].sum(1)[:, None]
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    res = pd.DataFrame({"차이 (b−a)": D.sum(0) / n.sum(), "95% 하한": lo, "95% 상한": hi}, index=cols)
    res["판정"] = np.select([res["95% 상한"] < 0, res["95% 하한"] > 0], ["좋아짐", "나빠짐"], "차이 불확실")
    return res
