import io
import glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import statsmodels.formula.api as smf
import pingouin as pg

st.set_page_config(page_title="Zernike 계수 분석", layout="wide")

REQUIRED_COLS = ["PID", "SEQ", "NO", "ORDER", "RADIAL", "NAME", "COEF"]


# ---------------------------------------------------------------------------
# Matplotlib 한글 폰트 설정
#   - 서버(Streamlit Cloud 등)에 설치된 한글 폰트를 자동 탐색해 등록.
#   - packages.txt 에 fonts-nanum 을 추가해두면 배포 환경에도 자동 설치됨.
# ---------------------------------------------------------------------------

def _setup_korean_font():
    preferred_names = ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "AppleGothic"]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in preferred_names:
        if name in installed:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name

    # 시스템에 폰트 파일은 있으나 matplotlib 캐시에 아직 없는 경우 직접 등록
    search_patterns = [
        "/usr/share/fonts/**/Nanum*.ttf",
        "/usr/share/fonts/**/Nanum*.otf",
        "/usr/share/fonts/**/NotoSansCJK*.ttc",
        "/usr/share/fonts/**/NotoSansKR*.otf",
        "/usr/share/fonts/**/malgun*.ttf",
    ]
    for pattern in search_patterns:
        for path in glob.glob(pattern, recursive=True):
            try:
                fm.fontManager.addfont(path)
                name = fm.FontProperties(fname=path).get_name()
                plt.rcParams["font.family"] = name
                plt.rcParams["axes.unicode_minus"] = False
                return name
            except Exception:
                continue

    # 한글 폰트를 못 찾은 경우: 마이너스 기호 깨짐만이라도 방지
    plt.rcParams["axes.unicode_minus"] = False
    return None


KOREAN_FONT_NAME = _setup_korean_font()

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
    "고위수차 (Total HOA, 3~7차)": {"orders": [3, 4, 5, 6, 7], "radial_abs": None},
    "3차항 고위수차 (3rd-order HOA)": {"orders": [3], "radial_abs": None},
    "4차항 고위수차 (4th-order HOA)": {"orders": [4], "radial_abs": None},
    "코마 (Coma, 3차·Radial ±1)": {"orders": [3], "radial_abs": 1},
    "세조각수차 (Trefoil, 3차·Radial ±3)": {"orders": [3], "radial_abs": 3},
    "구면수차 (Spherical Aberration, 4차·Radial 0)": {"orders": [4], "radial_abs": 0},
}

# 지표별 계산식 (화면 상단에 표시용)
COMPONENT_FORMULAS = {
    "고위수차 (Total HOA, 3~7차)": r"HOA_{total} = \sqrt{\sum_{Order=3}^{7} \sum_{Radial} C_{Order,Radial}^{\,2}}",
    "3차항 고위수차 (3rd-order HOA)": r"HOA_{3rd} = \sqrt{\sum_{Radial} C_{3,Radial}^{\,2}} \; (No.6\!\sim\!9)",
    "4차항 고위수차 (4th-order HOA)": r"HOA_{4th} = \sqrt{\sum_{Radial} C_{4,Radial}^{\,2}} \; (No.10\!\sim\!14)",
    "코마 (Coma, 3차·Radial ±1)": r"Coma = \sqrt{C_{3,-1}^{\,2} + C_{3,+1}^{\,2}} \; (No.7,\ No.8)",
    "세조각수차 (Trefoil, 3차·Radial ±3)": r"Trefoil = \sqrt{C_{3,-3}^{\,2} + C_{3,+3}^{\,2}} \; (No.6,\ No.9)",
    "구면수차 (Spherical Aberration, 4차·Radial 0)": r"SA = \left| C_{4,0} \right| \; (No.12)",
}


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
# 반복도
#   - 평균, Sw(반복측정 표준편차), CV, RC: One-way random effects ANOVA
#     (statsmodels mixedlm, REML)
#   - ICC: pingouin.intraclass_corr 의 ICC1 (one-way random effects 모델과
#     동일한 가정을 쓰는 지표로, PID=targets, SEQ=raters)
# ---------------------------------------------------------------------------

def repeatability_analysis(values_df):
    """
    values_df: columns PID, SEQ, VALUE (한 기기의 반복측정 결과)
    반환: {"mean", "Sw", "RC", "CV(%)", "ICC"} 또는 None / {"error": ...}
    """
    d = values_df.dropna(subset=["VALUE"]).copy()
    d["PID"] = d["PID"].astype(str)
    n_subjects = d["PID"].nunique()
    n_obs = len(d)
    if n_subjects < 2 or n_obs < n_subjects + 1:
        return None

    # --- mixedlm (REML): 평균, Sw, CV, RC ---
    try:
        model = smf.mixedlm("VALUE ~ 1", data=d, groups=d["PID"])
        result = model.fit(reml=True)
    except Exception as e:
        return {"error": str(e)}

    sigma2 = max(float(result.scale), 1e-12)
    grand_mean = float(d["VALUE"].mean())
    sw = np.sqrt(sigma2)  # within-subject (repeatability) SD
    cv = (sw / grand_mean * 100) if grand_mean != 0 else np.nan
    rc = 1.96 * np.sqrt(2) * sw  # Repeatability Coefficient

    # --- pingouin ICC1 ---
    # pingouin 버전에 따라 Type 라벨이 "ICC1"(구버전) 또는 "ICC(1,1)"(신버전)으로 다르게 표기됨
    icc_value = None
    try:
        icc_table = pg.intraclass_corr(
            data=d, targets="PID", raters="SEQ", ratings="VALUE", nan_policy="omit"
        )
        icc_row = icc_table[icc_table["Type"].isin(["ICC1", "ICC(1,1)"])]
        if icc_row.empty:
            icc_row = icc_table[icc_table["Type"].astype(str).str.startswith("ICC1")]
        if not icc_row.empty:
            icc_value = round(float(icc_row["ICC"].values[0]), 3)
    except Exception:
        icc_value = None

    return {
        "mean": grand_mean,
        "Sw": sw,
        "RC": rc,
        "CV(%)": cv,
        "ICC": icc_value,
    }


# ---------------------------------------------------------------------------
# Bland-Altman
# ---------------------------------------------------------------------------

def pair_average(test_vals, ref_vals):
    t = test_vals.groupby("PID", as_index=False)["VALUE"].mean().rename(columns={"VALUE": "TEST"})
    r = ref_vals.groupby("PID", as_index=False)["VALUE"].mean().rename(columns={"VALUE": "REF"})
    merged = pd.merge(t, r, on="PID", how="inner")
    return merged


def pair_first_seq(test_vals, ref_vals):
    """각 PID의 가장 첫 번째(SEQ 최솟값) 측정값끼리만 짝지어 비교."""
    t_first = test_vals.loc[test_vals.groupby("PID")["SEQ"].idxmin()]
    r_first = ref_vals.loc[ref_vals.groupby("PID")["SEQ"].idxmin()]
    t = t_first[["PID", "VALUE"]].rename(columns={"VALUE": "TEST"})
    r = r_first[["PID", "VALUE"]].rename(columns={"VALUE": "REF"})
    merged = pd.merge(t, r, on="PID", how="inner")
    return merged


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
    ax.axhline(stats["mean_diff"], color="blue", linestyle="-", label=f"평균 차이 = {stats['mean_diff']:.4f}")
    ax.axhline(stats["loa_upper"], color="red", linestyle="--", label=f"+1.96 SD = {stats['loa_upper']:.4f}")
    ax.axhline(stats["loa_lower"], color="red", linestyle="--", label=f"-1.96 SD = {stats['loa_lower']:.4f}")
    ax.set_xlabel("시험기기·대조기기 평균")
    ax.set_ylabel("차이 (시험기기 - 대조기기)")
    ax.set_title(f"Bland-Altman Plot: {metric_name}")
    ax.legend(fontsize=8)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
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
    if KOREAN_FONT_NAME is None:
        st.warning(
            "그래프용 한글 폰트를 찾지 못했습니다. 배포 저장소 루트에 `packages.txt` 파일을 만들고 "
            "`fonts-nanum`을 한 줄 적어두면 재배포 시 자동 설치됩니다."
        )

if not test_file or not ref_file:
    st.info("좌측에서 시험기기와 대조기기 파일을 모두 업로드해주세요.")
    st.stop()

try:
    df_test = read_uploaded(test_file)
    df_ref = read_uploaded(ref_file)
except Exception as e:
    st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
    st.stop()

component_names = list(PRESET_COMPONENTS.keys())

st.sidebar.header("2. 계산할 지표 선택")
selected_metric = st.sidebar.selectbox("지표 선택", component_names, index=0)
spec = PRESET_COMPONENTS[selected_metric]

tab1, tab2, tab3 = st.tabs(["① PID·SEQ별 계수", "② 반복도 (ANOVA)", "③ 일치도 (Bland-Altman)"])

# --------------------------- Tab 1 ---------------------------
with tab1:
    st.subheader(f"선택 지표: {selected_metric}")
    st.latex(COMPONENT_FORMULAS[selected_metric])

    val_test = compute_component(df_test, spec).rename(columns={"VALUE": "Coefficient"})
    val_ref = compute_component(df_ref, spec).rename(columns={"VALUE": "Coefficient"})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**시험기기**")
        st.dataframe(val_test.sort_values(["PID", "SEQ"]), use_container_width=True)
    with c2:
        st.markdown("**대조기기**")
        st.dataframe(val_ref.sort_values(["PID", "SEQ"]), use_container_width=True)

    combined = pd.merge(
        val_test.rename(columns={"Coefficient": "Coefficient_시험기기"}),
        val_ref.rename(columns={"Coefficient": "Coefficient_대조기기"}),
        on=["PID", "SEQ"],
        how="outer",
    ).sort_values(["PID", "SEQ"])
    csv_buf = io.StringIO()
    combined.to_csv(csv_buf, index=False)
    st.download_button("결과 CSV 다운로드 (시험/대조 병합)", csv_buf.getvalue(), file_name=f"{selected_metric}_by_PID_SEQ.csv")

# --------------------------- Tab 2 ---------------------------
with tab2:
    st.subheader(f"반복도 분석 (One-way random effects ANOVA, REML): {selected_metric}")
    val_test = compute_component(df_test, spec)
    val_ref = compute_component(df_ref, spec)

    res_test = repeatability_analysis(val_test)
    res_ref = repeatability_analysis(val_ref)

    def show_result(name, res):
        st.markdown(f"**{name}**")
        if res is None:
            st.warning("환자 수 또는 반복측정 횟수가 부족하여 계산할 수 없습니다.")
            return
        if "error" in res:
            st.error(f"모델 적합 실패: {res['error']}")
            return
        r = {
            "평균 (Mean)": round(res["mean"], 5),
            "Sw (반복측정 표준편차)": round(res["Sw"], 5),
            "RC (Repeatability Coefficient)": round(res["RC"], 5),
            "CV (%)": round(res["CV(%)"], 3) if not np.isnan(res["CV(%)"]) else None,
            "ICC (ICC1)": res["ICC"] if res["ICC"] is not None else "계산 불가",
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
        ["환자별 반복측정 평균끼리 비교", "첫 번째 측정값끼리 짝지어 비교"],
        horizontal=True,
    )

    val_test = compute_component(df_test, spec)
    val_ref = compute_component(df_ref, spec)

    if pairing_mode.startswith("환자별"):
        merged = pair_average(val_test, val_ref)
    else:
        merged = pair_first_seq(val_test, val_ref)

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
