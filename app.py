import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

st.set_page_config(page_title="Zernike 계수 분석", layout="wide")

REQUIRED_COLS = ["PID", "SEQ", "NO", "ORDER", "RADIAL", "NAME", "COEF"]

# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------

def read_uploaded(file):
    if file.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        # utf-8-sig: 엑셀에서 저장한 CSV에 흔한 BOM(\ufeff)을 자동 제거
        try:
            df = pd.read_csv(file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding="cp949")

    # 컬럼명 정리: BOM, 공백 제거 후 대문자로 통일
    df.columns = [c.replace("\ufeff", "").strip().upper() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"다음 컬럼이 없습니다: {missing} (현재 컬럼: {list(df.columns)})")

    df = df[REQUIRED_COLS].copy()

    # PID는 항상 문자열로 통일 (파일마다 숫자/문자로 다르게 읽히는 것을 방지 -> merge 오류의 주 원인)
    df["PID"] = df["PID"].astype(str).str.strip()

    # SEQ, NO, ORDER, RADIAL은 정수로 통일 (파일 간 dtype이 달라지면 merge 시 ValueError 발생)
    for c in ["SEQ", "NO", "ORDER", "RADIAL"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    df["COEF"] = pd.to_numeric(df["COEF"], errors="coerce")

    bad_rows = df[df[["SEQ", "NO", "ORDER", "RADIAL", "COEF"]].isna().any(axis=1)]
    if len(bad_rows) > 0:
        st.warning(f"숫자로 변환할 수 없는 값이 있는 행 {len(bad_rows)}개는 제외하고 진행합니다.")
        df = df.dropna(subset=["SEQ", "NO", "ORDER", "RADIAL", "COEF"])

    return df


# ---------------------------------------------------------------------------
# 지표(컴포넌트) 정의
#   각 지표는 (order 리스트, radial_abs 또는 None) 조합으로 정의.
#   radial_abs가 None이면 해당 order 전체(모든 radial)를 RMS로 묶음.
#   radial_abs가 숫자면 |RADIAL| == radial_abs 인 항만 묶음 (0이면 대칭항 단독).
# ---------------------------------------------------------------------------

PRESET_COMPONENTS = {
    "Total HOA (3~7차)": {"orders": [3, 4, 5, 6, 7], "radial_abs": None},
    "3rd-order HOA": {"orders": [3], "radial_abs": None},
    "4th-order HOA": {"orders": [4], "radial_abs": None},
    "5th-order HOA": {"orders": [5], "radial_abs": None},
    "6th-order HOA": {"orders": [6], "radial_abs": None},
    "7th-order HOA": {"orders": [7], "radial_abs": None},
    "Coma (3차, Radial ±1)": {"orders": [3], "radial_abs": 1},
    "Trefoil (3차, Radial ±3)": {"orders": [3], "radial_abs": 3},
    "Spherical Aberration (4차, Radial 0)": {"orders": [4], "radial_abs": 0},
    "Secondary Astigmatism (4차, Radial ±2)": {"orders": [4], "radial_abs": 2},
    "Quadrafoil (4차, Radial ±4)": {"orders": [4], "radial_abs": 4},
    "Astigmatism (2차, Radial ±2)": {"orders": [2], "radial_abs": 2},
    "Defocus (2차, Radial 0)": {"orders": [2], "radial_abs": 0},
    "Tilt (1차, Radial ±1)": {"orders": [1], "radial_abs": 1},
}


def build_individual_components(df):
    """No 1~35 개별 항목을 지표 사전에 추가 (NAME 기준, No로 구분)."""
    comps = {}
    lookup = df[["NO", "ORDER", "RADIAL", "NAME"]].drop_duplicates().sort_values("NO")
    for _, row in lookup.iterrows():
        label = f"No{int(row['NO'])}. {row['NAME']} (Order {int(row['ORDER'])}, Radial {int(row['RADIAL'])})"
        comps[label] = {"no": int(row["NO"])}
    return comps


def compute_component(df, spec):
    """spec에 따라 PID, SEQ 별로 지표 값을 계산해 반환. columns: PID, SEQ, VALUE"""
    if "no" in spec:
        sub = df[df["NO"] == spec["no"]]
        out = sub.groupby(["PID", "SEQ"], as_index=False)["COEF"].first()
        out = out.rename(columns={"COEF": "VALUE"})
        return out

    sub = df[df["ORDER"].isin(spec["orders"])]
    if spec["radial_abs"] is not None:
        sub = sub[sub["RADIAL"].abs() == spec["radial_abs"]]

    out = (
        sub.assign(SQ=sub["COEF"] ** 2)
        .groupby(["PID", "SEQ"], as_index=False)["SQ"]
        .sum()
    )
    out["VALUE"] = np.sqrt(out["SQ"])
    return out[["PID", "SEQ", "VALUE"]]


# ---------------------------------------------------------------------------
# 반복도 (One-way random effects ANOVA, REML via mixedlm)
# ---------------------------------------------------------------------------

def repeatability_mixedlm(values_df):
    """
    values_df: columns PID, SEQ, VALUE (한 기기의 반복측정 결과)
    반환: dict of 지표들
    """
    d = values_df.dropna(subset=["VALUE"]).copy()
    d["PID"] = d["PID"].astype(str)
    n_subjects = d["PID"].nunique()
    n_obs = len(d)
    if n_subjects < 2 or n_obs < n_subjects + 1:
        return None

    try:
        model = smf.mixedlm("VALUE ~ 1", data=d, groups=d["PID"])
        result = model.fit(reml=True)
    except Exception as e:
        return {"error": str(e)}

    tau2 = float(result.cov_re.iloc[0, 0])
    sigma2 = float(result.scale)
    grand_mean = float(d["VALUE"].mean())

    tau2 = max(tau2, 0.0)
    sigma2 = max(sigma2, 1e-12)

    sw = np.sqrt(sigma2)  # within-subject (repeatability) SD
    icc = tau2 / (tau2 + sigma2) if (tau2 + sigma2) > 0 else np.nan
    cv = (sw / grand_mean * 100) if grand_mean != 0 else np.nan
    rc = 1.96 * np.sqrt(2) * sw  # Repeatability Coefficient

    return {
        "n_subjects": n_subjects,
        "n_obs": n_obs,
        "grand_mean": grand_mean,
        "between_subject_var": tau2,
        "within_subject_var": sigma2,
        "within_subject_sd": sw,
        "ICC": icc,
        "CV(%)": cv,
        "Repeatability_Coefficient": rc,
    }


# ---------------------------------------------------------------------------
# Bland-Altman
# ---------------------------------------------------------------------------

def pair_average(test_vals, ref_vals):
    t = test_vals.groupby("PID", as_index=False)["VALUE"].mean().rename(columns={"VALUE": "TEST"})
    r = ref_vals.groupby("PID", as_index=False)["VALUE"].mean().rename(columns={"VALUE": "REF"})
    merged = pd.merge(t, r, on="PID", how="inner")
    return merged


def pair_by_seq(test_vals, ref_vals):
    t = test_vals.rename(columns={"VALUE": "TEST"})
    r = ref_vals.rename(columns={"VALUE": "REF"})
    merged = pd.merge(t, r, on=["PID", "SEQ"], how="inner")
    return merged[["PID", "SEQ", "TEST", "REF"]]


def bland_altman_stats(merged):
    d = merged.copy()
    d["MEAN"] = (d["TEST"] + d["REF"]) / 2
    d["DIFF"] = d["TEST"] - d["REF"]
    mean_diff = d["DIFF"].mean()
    sd_diff = d["DIFF"].std(ddof=1)
    loa_upper = mean_diff + 1.96 * sd_diff
    loa_lower = mean_diff - 1.96 * sd_diff
    stats = {
        "n_pairs": len(d),
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "loa_upper": loa_upper,
        "loa_lower": loa_lower,
    }
    return d, stats


def plot_bland_altman(d, stats, metric_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(d["MEAN"], d["DIFF"], alpha=0.7, edgecolor="k", linewidth=0.3)
    ax.axhline(stats["mean_diff"], color="blue", linestyle="-", label=f"Mean diff = {stats['mean_diff']:.4f}")
    ax.axhline(stats["loa_upper"], color="red", linestyle="--", label=f"+1.96 SD = {stats['loa_upper']:.4f}")
    ax.axhline(stats["loa_lower"], color="red", linestyle="--", label=f"-1.96 SD = {stats['loa_lower']:.4f}")
    ax.set_xlabel("Mean of Test & Reference")
    ax.set_ylabel("Difference (Test - Reference)")
    ax.set_title(f"Bland-Altman Plot: {metric_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("Zernike 계수 분석 대시보드")
st.caption("시험기기 / 대조기기 반복측정 Zernike 데이터를 업로드하여 계수 계산, 반복도, 일치도 분석을 수행합니다.")

with st.sidebar:
    st.header("1. 데이터 업로드")
    test_file = st.file_uploader("시험기기 파일 (csv/xlsx)", type=["csv", "xlsx", "xls"], key="test")
    ref_file = st.file_uploader("대조기기 파일 (csv/xlsx)", type=["csv", "xlsx", "xls"], key="ref")
    st.caption("필수 컬럼: PID, SEQ, NO, ORDER, RADIAL, NAME, COEF")

if not test_file or not ref_file:
    st.info("좌측에서 시험기기와 대조기기 파일을 모두 업로드해주세요.")
    st.stop()

try:
    df_test = read_uploaded(test_file)
    df_ref = read_uploaded(ref_file)
except Exception as e:
    st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
    st.stop()

# 지표 선택 목록 구성 (두 파일 공통 NO 기준으로 개별 항목 생성)
individual_test = build_individual_components(df_test)
all_components = {**PRESET_COMPONENTS, **individual_test}
component_names = list(all_components.keys())

st.sidebar.header("2. 계산할 지표 선택")
selected_metric = st.sidebar.selectbox("지표 선택", component_names, index=0)
spec = all_components[selected_metric]

tab1, tab2, tab3 = st.tabs(["① PID·SEQ별 계수", "② 반복도 (ANOVA)", "③ 일치도 (Bland-Altman)"])

# --------------------------- Tab 1 ---------------------------
with tab1:
    st.subheader(f"선택 지표: {selected_metric}")
    val_test = compute_component(df_test, spec).rename(columns={"VALUE": "TEST"})
    val_ref = compute_component(df_ref, spec).rename(columns={"VALUE": "REF"})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**시험기기**")
        st.dataframe(val_test.sort_values(["PID", "SEQ"]), use_container_width=True)
    with c2:
        st.markdown("**대조기기**")
        st.dataframe(val_ref.sort_values(["PID", "SEQ"]), use_container_width=True)

    combined = pd.merge(val_test, val_ref, on=["PID", "SEQ"], how="outer").sort_values(["PID", "SEQ"])
    csv_buf = io.StringIO()
    combined.to_csv(csv_buf, index=False)
    st.download_button("결과 CSV 다운로드 (시험/대조 병합)", csv_buf.getvalue(), file_name=f"{selected_metric}_by_PID_SEQ.csv")

# --------------------------- Tab 2 ---------------------------
with tab2:
    st.subheader(f"반복도 분석 (One-way random effects ANOVA, REML): {selected_metric}")
    val_test = compute_component(df_test, spec)
    val_ref = compute_component(df_ref, spec)

    res_test = repeatability_mixedlm(val_test)
    res_ref = repeatability_mixedlm(val_ref)

    def show_result(name, res):
        st.markdown(f"**{name}**")
        if res is None:
            st.warning("환자 수 또는 반복측정 횟수가 부족하여 계산할 수 없습니다.")
            return
        if "error" in res:
            st.error(f"모델 적합 실패: {res['error']}")
            return
        r = {
            "환자 수": res["n_subjects"],
            "총 관측치 수": res["n_obs"],
            "전체 평균": round(res["grand_mean"], 5),
            "환자간 분산 (τ²)": round(res["between_subject_var"], 6),
            "환자내(오차) 분산 (σ²)": round(res["within_subject_var"], 6),
            "반복측정 표준편차 (Sw)": round(res["within_subject_sd"], 5),
            "ICC": round(res["ICC"], 4),
            "CV (%)": round(res["CV(%)"], 3) if not np.isnan(res["CV(%)"]) else None,
            "Repeatability Coefficient (1.96*√2*Sw)": round(res["Repeatability_Coefficient"], 5),
        }
        st.table(pd.DataFrame(r.items(), columns=["항목", "값"]).set_index("항목"))

    c1, c2 = st.columns(2)
    with c1:
        show_result("시험기기", res_test)
    with c2:
        show_result("대조기기", res_ref)

# --------------------------- Tab 3 ---------------------------
with tab3:
    st.subheader(f"일치도 분석 (Bland-Altman): {selected_metric}")
    pairing_mode = st.radio(
        "짝짓기 방식 선택",
        ["환자별 반복측정 평균끼리 비교", "동일 SEQ(순서)끼리 짝지어 비교"],
        horizontal=True,
    )

    val_test = compute_component(df_test, spec)
    val_ref = compute_component(df_ref, spec)

    if pairing_mode.startswith("환자별"):
        merged = pair_average(val_test, val_ref)
    else:
        merged = pair_by_seq(val_test, val_ref)

    if merged.empty:
        st.warning("짝지을 수 있는 데이터가 없습니다. PID(및 SEQ)가 두 파일 간에 일치하는지 확인해주세요.")
    else:
        d, stats = bland_altman_stats(merged)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**요약 통계**")
            summary = {
                "짝(pair) 수": stats["n_pairs"],
                "평균 차이 (Test-Ref)": round(stats["mean_diff"], 5),
                "차이의 표준편차": round(stats["sd_diff"], 5),
                "상한 일치한계 (+1.96SD)": round(stats["loa_upper"], 5),
                "하한 일치한계 (-1.96SD)": round(stats["loa_lower"], 5),
            }
            st.table(pd.DataFrame(summary.items(), columns=["항목", "값"]).set_index("항목"))
            csv_buf2 = io.StringIO()
            d.to_csv(csv_buf2, index=False)
            st.download_button("짝지은 데이터 CSV 다운로드", csv_buf2.getvalue(), file_name=f"{selected_metric}_bland_altman_pairs.csv")
        with c2:
            fig = plot_bland_altman(d, stats, selected_metric)
            st.pyplot(fig)

        st.markdown("**짝지은 데이터**")
        st.dataframe(d, use_container_width=True)
