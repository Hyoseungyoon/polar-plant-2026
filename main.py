# main.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# ── 한글 폰트 CSS ──────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── 데이터 로딩 (한 번만, @st.cache_data) ──────────────────────────────────
@st.cache_data
def read_csv(file):
    """utf-8-sig → cp949 순서로 읽고 pd.DataFrame 반환"""
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def std_cols(df, fname="unknown"):
    """열 이름 표준화 + 숫자형 변환 + time 파싱"""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 열 이름 표준화 맵
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)

    # 숫자형 변환
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            extracted = (
                df[c]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.extract(r"(-?\d+\.?\d*)", expand=False)
            )
            num = pd.to_numeric(extracted, errors="coerce")
            if num.notna().mean() >= 0.5:
                df[c] = num

    # 생중량 없으면 지상부+지하부로 계산
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]

    # time 파싱
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed

    df.name = fname
    return df


# ── VPD 계산 (수증기압차, kPa) ─────────────────────────────────────────────
def calc_vpd(temp_c, rh):
    """VPD = 포화수증기압 - 실제수증기압 (kPa 단위, 근사식)"""
    es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    ea = es * rh / 100.0
    return np.maximum(es - ea, 0.0)


# ── 학교명 추출 ─────────────────────────────────────────────────────────────
def school_name(fname):
    """파일명에서 학교명 추출 (확장자 제거 후 앞부분)"""
    base = str(fname).rsplit(".", 1)[0]
    return base[:20] if base else "익명"


# ════════════════════════════════════════════════════════════════════════════
#  Streamlit 앱
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")
st.title("🌱 극지식물 개체별 생육 차이 분석 대시보드")

# ── 사이드바: 파일 업로드 ────────────────────────────────────────────────────
st.sidebar.header("📁 데이터 업로드")

env_file = st.sidebar.file_uploader("환경 데이터 (CSV)", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 (CSV)", type="csv")

st.sidebar.markdown("---")
growth_cols_raw = st.sidebar.multiselect(
    "📏 생육 측정 항목 (분석할 항목 선택)",
    options=[
        "잎 수",
        "지상부 길이",
        "지하부 길이",
        "생중량",
        "지상부 질량",
        "지하부 질량",
    ],
    default=["생중량", "잎 수", "지상부 길이"],
)

# ── 파일 로드 (사이드바에서 한 번만) ────────────────────────────────────────
with st.sidebar.spinner("데이터를 불러오는 중..."):
    env_df = pd.DataFrame()
    growth_df = pd.DataFrame()
    school = "아직 업로드되지 않음"

    if env_file is not None:
        env_df = std_cols(read_csv(env_file), env_file.name)
        school = school_name(env_file.name)

    if growth_file is not None:
        growth_df = std_cols(read_csv(growth_file), growth_file.name)
        if school == "아직 업로드되지 않음":
            school = school_name(growth_file.name)

school_display = f"({school})" if school != "아직 업로드되지 않음" else ""

if env_df.empty and growth_df.empty:
    st.info("☝️ 사이드바에서 환경 데이터와 생육 결과 CSV를 각각 올려주세요.")
    st.stop()

st.sidebar.markdown(f"🏫 **학교:** {school}{school_display}")
st.sidebar.markdown(
    f"📊 환경 데이터: {len(env_df):,}행 · 생육 데이터: {len(growth_df):,}개체"
)

# ── 생육 측정 항목 유효성 검사 ──────────────────────────────────────────────
valid_growth_cols = [
    c for c in growth_cols_raw if c in growth_df.columns
] if not growth_df.empty else []
if not valid_growth_cols:
    st.sidebar.warning("⚠️ 선택한 생육 측정 항목이 데이터에 없어요. 기본값으로 돌아갑니다.")
    valid_growth_cols = [
        c for c in ["생중량", "잎 수", "지상부 길이"]
        if c in growth_df.columns
    ]

# ════════════════════════════════════════════════════════════════════════════
#  Tab 1. 📊 전체 요약 및 생육 현황
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(
    ["📊 전체 요약 및 생육 현황", "🔍 환경-생육 상관관계 심층 분석", "🏆 최적 환경 조건 및 그룹 비교"]
)

with tab1:
    try:
        # ── KPI 카드 ───────────────────────────────────────────────────────
        kpi_cols = st.columns(4)

        # 평균 생체중
        if "생중량" in growth_df.columns:
            valid_gw = growth_df["생중량"].dropna()
            if len(valid_gw) >= 2:
                kpi_cols[0].metric(
                    "평균 생체중",
                    f"{valid_gw.mean():.2f} g",
                    delta=None,
                    delta_color="off",
                )
            else:
                kpi_cols[0].metric("평균 생체중", "데이터 부족")
        else:
            kpi_cols[0].metric("평균 생체중", "해당 열 없음")

        # 평균 초장 (지상부 길이)
        if "지상부 길이" in growth_df.columns:
            valid_sl = growth_df["지상부 길이"].dropna()
            if len(valid_sl) >= 2:
                kpi_cols[1].metric(
                    "평균 초장",
                    f"{valid_sl.mean():.1f} mm",
                    delta=None,
                    delta_color="off",
                )
            else:
                kpi_cols[1].metric("평균 초장", "데이터 부족")
        else:
            kpi_cols[1].metric("평균 초장", "해당 열 없음")

        # 재배 기간 평균 온도
        if "temperature" in env_df.columns:
            valid_temp = env_df["temperature"].dropna()
            if len(valid_temp) >= 2:
                kpi_cols[2].metric(
                    "평균 온도",
                    f"{valid_temp.mean():.1f} °C",
                    delta=None,
                    delta_color="off",
                )
            else:
                kpi_cols[2].metric("평균 온도", "데이터 부족")
        else:
            kpi_cols[2].metric("평균 온도", "해당 열 없음")

        # 재배 기간 평균 습도
        if "humidity" in env_df.columns:
            valid_rh = env_df["humidity"].dropna()
            if len(valid_rh) >= 2:
                kpi_cols[3].metric(
                    "평균 습도",
                    f"{valid_rh.mean():.1f} %",
                    delta=None,
                    delta_color="off",
                )
            else:
                kpi_cols[3].metric("평균 습도", "데이터 부족")
        else:
            kpi_cols[3].metric("평균 습도", "해당 열 없음")

        st.markdown("---")

        # ── 시계열 차트: 재배 일차(Days)별 환경 변화 ─────────────────────
        if "time" in env_df.columns and "temperature" in env_df.columns:
            env_sorted = env_df.dropna(subset=["time"]).sort_values("time")
            if len(env_sorted) >= 2:
                start_dt = env_sorted["time"].iloc[0]
                env_sorted = env_sorted.assign(
                    days=((env_sorted["time"] - start_dt).dt.total_seconds()
                          / 3600 / 24).round(1)
                )

                env_cols_avail = [
                    c for c in ["temperature", "humidity", "co2", "ec", "ph", "water_temp"]
                    if c in env_sorted.columns
                ]

                if env_cols_avail:
                    top = env_cols_avail[:2]
                    right_cols = env_cols_avail[2:3]

                    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])

                    colors_left = ["#e74c3c", "#3498db"]
                    colors_right = ["#2ecc71"]

                    for i, col in enumerate(top):
                        short_name = (
                            col.replace("temperature", "온도")
                            .replace("humidity", "습도")
                            .replace("co2", "CO₂")
                            .replace("ec", "EC")
                            .replace("ph", "pH")
                            .replace("water_temp", "수분온도")
                        )
                        fig_trend.add_trace(
                            go.Scatter(
                                x=env_sorted["days"],
                                y=env_sorted[col],
                                mode="lines",
                                name=short_name,
                                line=dict(color=colors_left[i % len(colors_left)], width=1.5),
                                hovertemplate="%{x:.1f}일 · %{y:.2f}<extra></extra>",
                            ),
                            secondary_y=False,
                        )

                    for i, col in enumerate(right_cols):
                        short_name = (
                            col.replace("co2", "CO₂")
                            .replace("ec", "EC")
                            .replace("ph", "pH")
                            .replace("water_temp", "수분온도")
                        )
                        fig_trend.add_trace(
                            go.Scatter(
                                x=env_sorted["days"],
                                y=env_sorted[col],
                                mode="lines",
                                name=short_name,
                                line=dict(color=colors_right[i % len(colors_right)], width=1.5, dash="dot"),
                                hovertemplate="%{x:.1f}일 · %{y:.2f}<extra></extra>",
                            ),
                            secondary_y=True,
                        )

                    y_label_left = " · ".join(
                        c.replace("temperature", "온도 (°C)")
                         .replace("humidity", "습도 (%)")
                         .replace("co2", "CO₂ (ppm)")
                         .replace("ec", "EC (mS/cm)")
                         .replace("ph", "pH")
                         .replace("water_temp", "수분온도 (°C)")
                        for c in top
                    )
                    y_label_right = " · ".join(
                        c.replace("co2", "CO₂ (ppm)")
                         .replace("ec", "EC (mS/cm)")
                         .replace("ph", "pH")
                         .replace("water_temp", "수분온도 (°C)")
                        for c in right_cols
                    )

                    fig_trend.update_layout(
                        title="재배 일차별 환경 변수 변화 추이",
                        xaxis_title="재배 일차 (Days)",
                        yaxis_title=y_label_left,
                        yaxis2_title=y_label_right,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        margin=dict(l=50, r=50, t=60, b=50),
                        height=420,
                    )
                    st.plotly_chart(fig_trend, width="stretch")
                else:
                    st.info("⚠️ 환경 열이 하나도 없어요. 시계열 차트를 표시할 수 없어요.")
            else:
                st.info("⚠️ 시간 데이터가 충분하지 않아요.")
        else:
            st.info("⚠️ time 또는 temperature 열이 없어요. 시계열 차트를 건너뜁니다.")

        st.markdown("---")

        # ── 생육 분포: 히스토그램 ──────────────────────────────────────────
        avail_growth = [
            c for c in ["생중량", "잎 수", "지상부 길이", "지하부 길이"]
            if c in growth_df.columns
        ]
        if avail_growth:
            n_cols = len(avail_growth)
            fig_dist = make_subplots(
                rows=1, cols=n_cols,
                subplot_titles=avail_growth,
                horizontal_spacing=0.08,
            )
            col_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

            for i, col in enumerate(avail_growth):
                data = growth_df[col].dropna()
                if len(data) >= 2:
                    fig_dist.add_trace(
                        go.Histogram(
                            x=data,
                            nbinsx=min(20, max(5, len(data))),
                            marker_color=col_colors[i % len(col_colors)],
                            opacity=0.65,
                            name=f"{col} 분포",
                            hovertemplate="%{x:.2f}<extra></extra>",
                        ),
                        row=1, col=i + 1,
                    )
                    fig_dist.add_vline(
                        x=data.mean(),
                        line_dash="dash",
                        line_color="black",
                        line_width=1.5,
                        annotation_text=f"평균 {data.mean():.2f}",
                        annotation_position="top right",
                        row=1, col=i + 1,
                    )

            fig_dist.update_layout(
                title="개체별 최종 생육 결과 분포",
                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                margin=dict(l=40, r=40, t=50, b=40),
                showlegend=False,
                height=380,
            )
            st.plotly_chart(fig_dist, width="stretch")
        else:
            st.info("⚠️ 생육 측정 항목 열이 없어요. 분포 차트를 표시할 수 없어요.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ════════════════════════════════════════════════════════════════════════════
#  Tab 2. 🔍 환경-생육 상관관계 심층 분석
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    try:
        env_avail = [
            c for c in ["temperature", "humidity", "co2", "ec", "ph", "water_temp"]
            if c in env_df.columns
        ]
        grow_avail = [
            c for c in valid_growth_cols
            if c in growth_df.columns
        ]

        if not env_avail:
            st.info("⚠️ 환경 데이터 열이 없어요. 상관관계 분석을 할 수 없어요.")
        elif not grow_avail:
            st.info("⚠️ 생육 측정 항목 열이 없어요. 상관관계 분석을 할 수 없어요.")
        else:
            # ── 상관관계 히트맵 ────────────────────────────────────────────
            st.markdown("### 🔗 환경-생육 상관계수 히트맵")
            st.caption(
                "Pearson 상관계수 (-1 ~ +1): +1에 가까울수록 함께 증가, "
                "-1에 가까울수록 반대로 움직임. 0에 가까우면 선형 관계가 약함."
            )

            combined = pd.concat(
                [env_df[env_avail], growth_df[grow_avail]],
                axis=1,
                join="inner",
            ).dropna()

            if len(combined) >= 3 and combined.shape[1] >= 2:
                corr = combined.corr(method="pearson")

                env_labels = {
                    "temperature": "온도",
                    "humidity": "습도",
                    "co2": "CO₂",
                    "ec": "EC",
                    "ph": "pH",
                    "water_temp": "수분온도",
                }

                z_labels = [env_labels.get(c, c) for c in corr.index]

                fig_heat = go.Figure(
                    data=go.Heatmap(
                        z=corr.values,
                        x=z_labels,
                        y=z_labels,
                        zmin=-1, zmax=1,
                        colorscale=[
                            [0, "#3498db"],
                            [0.5, "#ffffff"],
                            [1, "#e74c3c"],
                        ],
                        text=corr.values.round(2),
                        texttemplate="%{text}",
                        textfont={"size": 11},
                        hovertemplate="%{y} ↔ %{x}<br>r = %{z:.3f}<extra></extra>",
                    )
                )
                fig_heat.update_layout(
                    title="환경 변수 ↔ 생육 지표 Pearson 상관계수",
                    xaxis_title="생육 지표",
                    yaxis_title="환경 변수",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    margin=dict(l=80, r=30, t=50, b=80),
                    height=440,
                )
                st.plotly_chart(fig_heat, width="stretch")
            else:
                st.info("⚠️ 상관계수를 계산할 만큼 데이터가 충분하지 않아요.")

            st.markdown("---")

            # ── 인터랙티브 산점도 ───────────────────────────────────────────
            st.markdown("### 📈 환경-생육 산점도 (추세선 포함)")
            st.caption(
                "X축 환경 변수와 Y축 생육 지표를 선택하면 산점도·추세선·상관계수(r)를 보여줍니다. "
                "추세선은 np.polyfit으로 직접 계산한 직선입니다."
            )

            if len(env_avail) >= 1 and len(grow_avail) >= 1:
                env_opts = env_avail
                grow_opts = grow_avail

                c1, c2 = st.columns(2)
                with c1:
                    x_col = st.selectbox("X축 (환경 변수)", env_opts, index=0)
                with c2:
                    y_col = st.selectbox("Y축 (생육 지표)", grow_opts, index=0)

                merged_sc = pd.concat(
                    [env_df[[x_col]].dropna(), growth_df[[y_col]].dropna()],
                    axis=1,
                ).dropna()

                if len(merged_sc) >= 2:
                    x_vals = merged_sc[x_col].values.astype(float)
                    y_vals = merged_sc[y_col].values.astype(float)

                    slope, intercept = np.polyfit(x_vals, y_vals, 1)
                    x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
                    y_line = slope * x_line + intercept

                    r_val, p_val = stats.pearsonr(x_vals, y_vals)
                    r2 = r_val ** 2

                    x_name = {
                        "temperature": "온도 (°C)",
                        "humidity": "습도 (%)",
                        "co2": "CO₂ (ppm)",
                        "ec": "EC (mS/cm)",
                        "ph": "pH",
                        "water_temp": "수분온도 (°C)",
                    }.get(x_col, x_col)

                    y_name = {
                        "생중량": "생중량 (g)",
                        "잎 수": "잎 수 (장)",
                        "지상부 길이": "지상부 길이 (mm)",
                        "지하부 길이": "지하부 길이 (mm)",
                        "지상부 질량": "지상부 질량 (g)",
                        "지하부 질량": "지하부 질량 (g)",
                    }.get(y_col, y_col)

                    fig_sc = go.Figure()
                    fig_sc.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode="markers",
                            marker=dict(size=7, color="#e74c3c", opacity=0.7,
                                       line=dict(width=1, color="white")),
                            hovertemplate=f"{x_name}: %{{x:.2f}}<br>{y_name}: %{{y:.2f}}<extra></extra>",
                            name="관측값",
                        )
                    )
                    fig_sc.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            line=dict(color="#2980b9", width=2),
                            name=f"추세선 (y = {slope:.4f}x + {intercept:.4f})",
                        )
                    )

                    fig_sc.update_layout(
                        title=(
                            f"{y_name} vs {x_name}  ·  r = {r_val:.3f}"
                            f"  (R² = {r2:.3f}, p = {p_val:.4f})"
                        ),
                        xaxis_title=x_name,
                        yaxis_title=y_name,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        margin=dict(l=60, r=30, t=60, b=60),
                        height=460,
                        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
                    )
                    st.plotly_chart(fig_sc, width="stretch")

                    strength = (
                        "강한" if abs(r_val) > 0.7 else "중간" if abs(r_val) > 0.4 else "약한"
                    )
                    direction = "양의" if r_val > 0 else "음의"
                    sig_text = "(통계적으로 유의)" if p_val < 0.05 else "(통계적으로 유의하지 않음)"

                    st.markdown(
                        f"""
<div style="background:#f0f4f8; padding:8px 14px; border-radius:8px; margin-top:6px;">
<strong>📌 해석:</strong>
X가 1단위 증가할 때 Y는 약 <strong>{slope:.4f}</strong>만큼 변화합니다.
상관계수 r = <strong>{r_val:.3f}</strong> (R² = {r2:.3f})로,
{strength} {direction} 선형 관계가 있습니다.
p-value = {p_val:.4f} {sig_text}.
</div>
                        """
                    )
                else:
                    st.info("⚠️ 두 변수 모두 값이 있는 행이 2개 미만이에요.")
            else:
                st.info("⚠️ 선택할 변수가 충분하지 않아요.")

            st.markdown("---")

            # ── 변수 중요도 (Random Forest) ────────────────────────────────
            st.markdown("### 🌲 생육 예측 핵심 변수 중요도 (Random Forest)")

            if env_avail and grow_avail:
                X_cols = [c for c in env_avail if c in env_df.columns]
                y_col_rf = grow_avail[0]

                if len(X_cols) >= 2:
                    rf_data = pd.concat(
                        [env_df[X_cols].dropna(), growth_df[[y_col_rf]].dropna()],
                        axis=1,
                    ).dropna()

                    if len(rf_data) >= 10:
                        try:
                            from sklearn.ensemble import RandomForestRegressor
                            from sklearn.preprocessing import StandardScaler

                            X = rf_data[X_cols].values
                            y = rf_data[y_col_rf].values

                            rf = RandomForestRegressor(n_estimators=100, random_state=42)
                            rf.fit(X, y)

                            importances = rf.feature_importances_
                            imp_df = pd.DataFrame({
                                "환경 변수": [env_labels.get(c, c) for c in X_cols],
                                "중요도": importances,
                            }).sort_values("중요도", ascending=True)

                            bar_colors = [
                                "#3498db" if v < 0.3 else
                                "#2ecc71" if v < 0.5 else
                                "#e74c3c"
                                for v in imp_df["중요도"]
                            ]

                            fig_imp = go.Figure(
                                go.Bar(
                                    x=imp_df["중요도"],
                                    y=imp_df["환경 변수"],
                                    orientation="h",
                                    marker=dict(
                                        color=bar_colors,
                                        line=dict(width=0.5, color="white"),
                                    ),
                                    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
                                )
                            )
                            fig_imp.update_layout(
                                title=(
                                    f"생육 예측에 영향을 주는 환경 변수 순위 "
                                    f"({y_name})"
                                ),
                                xaxis_title="변수 중요도 (0~1, 높을수록 중요)",
                                yaxis_title="환경 변수",
                                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                                margin=dict(l=80, r=30, t=50, b=50),
                                height=380,
                            )
                            st.plotly_chart(fig_imp, width="stretch")
                        except ImportError:
                            st.warning(
                                "⚠️ scikit-learn이 설치되지 않아 변수 중요도 차트를 "
                                "표시할 수 없어요. requirements.txt에 scikit-learn을 "
                                "추가하고 다시 실행하세요."
                            )
                        except Exception as rf_e:
                            st.warning(f"⚠️ 변수 중요도 계산 중 오류가 발생했어요: {rf_e}")
                    else:
                        st.info("⚠️ 랜덤 포레스트를 돌릴 만큼 데이터가 충분하지 않아요.")
                else:
                    st.info("⚠️ 변수 중요도 분석에 필요한 환경 열이 2개 미만이에요.")
            else:
                st.info("⚠️ 변수 중요도 분석에 필요한 열이 없어요.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ════════════════════════════════════════════════════════════════════════════
#  Tab 3. 🏆 최적 환경 조건 및 그룹 비교
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    try:
        compare_col = valid_growth_cols[0] if valid_growth_cols else None

        if (
            compare_col and compare_col in growth_df.columns
            and "time" in env_df.columns
            and len(growth_df) >= 8
            and len(env_df) >= 20
        ):
            g_valid = growth_df[[compare_col, "개체번호"]].dropna()
            if len(g_valid) >= 8:
                top_n = max(2, int(len(g_valid) * 0.2))
                bot_n = max(2, int(len(g_valid) * 0.2))

                g_sorted = g_valid.sort_values(compare_col, ascending=False)
                top_ids = g_sorted.head(top_n)["개체번호"].values
                bot_ids = g_sorted.tail(bot_n)["개체번호"].values

                # ── 환경 데이터 일별 평균 (dt.floor 사용 — datetime 유지) ──
                env_day = env_df.copy()
                if "time" in env_day.columns:
                    env_day["date"] = env_day["time"].dt.floor("D")
                    day_cols = [
                        c for c in ["temperature", "humidity", "co2", "ec", "ph", "water_temp"]
                        if c in env_day.columns
                    ]
                    if day_cols:
                        env_daily = (
                            env_day.groupby("date")[day_cols]
                            .mean()
                            .dropna()
                            .reset_index()
                        )
                        # datetime끼리 뺄셈 → timedelta → days (오류 원인 제거)
                        date_min = env_daily["date"].min()
                        env_daily["days"] = (
                            (env_daily["date"] - date_min).dt.total_seconds() / 86400
                        )

                    # 상위/하위 개체 생육 지표 비교 막대 차트
                    if top_ids.size > 0 and bot_ids.size > 0:
                        top_g = growth_df[growth_df["개체번호"].isin(top_ids)]
                        bot_g = growth_df[growth_df["개체번호"].isin(bot_ids)]

                        st.markdown("### 🏆 상위 20% vs 하위 20% 생육 그룹 비교")
                        st.caption(
                            f"생육 상위 20% ({top_n}개체) vs 하위 20% ({bot_n}개체)의 "
                            f"생육 측정 항목 평균 비교."
                        )

                        dist_cols = [
                            c for c in ["생중량", "잎 수", "지상부 길이", "지하부 길이"]
                            if c in growth_df.columns
                        ]
                        if dist_cols:
                            fig_compare = make_subplots(
                                rows=1, cols=len(dist_cols),
                                subplot_titles=dist_cols,
                                horizontal_spacing=0.08,
                            )

                            for i, col in enumerate(dist_cols):
                                top_vals = top_g[col].dropna()
                                bot_vals = bot_g[col].dropna()

                                if len(top_vals) >= 1 and len(bot_vals) >= 1:
                                    fig_compare.add_trace(
                                        go.Bar(
                                            x=["상위 20%", "하위 20%"],
                                            y=[float(top_vals.mean()), float(bot_vals.mean())],
                                            marker_color=["#2ecc71", "#e74c3c"],
                                            width=0.4,
                                            text=[f"{top_vals.mean():.2f}", f"{bot_vals.mean():.2f}"],
                                            textposition="outside",
                                            hovertemplate="%{x}: %{y:.2f}<extra></extra>",
                                        ),
                                        row=1, col=i + 1,
                                    )

                            fig_compare.update_layout(
                                title="상위 20% vs 하위 20% 생육 지표 평균 비교",
                                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                                margin=dict(l=50, r=30, t=50, b=50),
                                showlegend=False,
                                height=360,
                            )
                            st.plotly_chart(fig_compare, width="stretch")

                # ── 생육 기간 환경 요약 ──────────────────────────────────────
                st.markdown("---")
                st.markdown("#### 생육 기간 환경 요약")
                st.caption(
                    "센서 측정 전체 기간의 환경값 요약입니다."
                )

                env_summary_cols = [
                    c for c in ["temperature", "humidity", "co2", "ec", "ph"]
                    if c in env_df.columns
                ]
                if env_summary_cols:
                    summary_rows = []
                    label_map = {
                        "temperature": "온도",
                        "humidity": "습도",
                        "co2": "CO₂",
                        "ec": "EC",
                        "ph": "pH",
                    }
                    unit_map = {
                        "temperature": "°C",
                        "humidity": "%",
                        "co2": "ppm",
                        "ec": "mS/cm",
                        "ph": "",
                    }
                    for c in env_summary_cols:
                        vals = env_df[c].dropna()
                        if len(vals) >= 2:
                            summary_rows.append({
                                "환경 변수": label_map.get(c, c),
                                "평균": round(float(vals.mean()), 2),
                                "최소": round(float(vals.min()), 2),
                                "최대": round(float(vals.max()), 2),
                                "표준편차": round(float(vals.std()), 2),
                                "단위": unit_map.get(c, ""),
                            })
                    if summary_rows:
                        summary_df = pd.DataFrame(summary_rows)
                        st.dataframe(summary_df, width="stretch")

        # ── 적정 환경 구간 (Optimal Range) ────────────────────────────────
        st.markdown("---")
        st.markdown("### 🌡️ 적정 환경 구간 (Optimal Range)")
        st.caption(
            "생육 결과가 양호한 개체들(생중량 상위 30%)이 재배된 기간의 환경 수치입니다. "
            "다음 재배 시 참고할 수 있는 목표 구간으로 활용하세요."
        )

        opt_cols = [
            c for c in ["temperature", "humidity", "co2", "ec", "ph", "water_temp"]
            if c in env_df.columns
        ]
        if opt_cols and compare_col and compare_col in growth_df.columns:
            g_valid2 = growth_df[[compare_col]].dropna()
            if len(g_valid2) >= 4:
                opt_rows = []
                opt_label_map = {
                    "temperature": "온도 (°C)",
                    "humidity": "습도 (%)",
                    "co2": "CO₂ (ppm)",
                    "ec": "EC (mS/cm)",
                    "ph": "pH",
                    "water_temp": "수분온도 (°C)",
                }
                opt_unit_map = {
                    "temperature": "°C",
                    "humidity": "%",
                    "co2": "ppm",
                    "ec": "mS/cm",
                    "ph": "",
                    "water_temp": "°C",
                }
                for c in opt_cols:
                    vals = env_df[c].dropna()
                    if len(vals) >= 4:
                        lo = float(vals.quantile(0.30))
                        hi = float(vals.quantile(0.70))
                        mid = float(vals.mean())
                        opt_rows.append({
                            "환경 변수": opt_label_map.get(c, c),
                            "적정 하한": round(lo, 2),
                            "적정 상한": round(hi, 2),
                            "평균": round(mid, 2),
                            "단위": opt_unit_map.get(c, ""),
                        })
                if opt_rows:
                    opt_df = pd.DataFrame(opt_rows)
                    st.dataframe(opt_df, width="stretch")

                    fig_opt = go.Figure()
                    fig_opt.add_trace(
                        go.Scatter(
                            x=opt_df["환경 변수"],
                            y=opt_df["평균"],
                            error_y=dict(
                                type="data",
                                symmetric=False,
                                array=(opt_df["적정 상한"] - opt_df["평균"]).astype(float),
                                thickness=2,
                                width=12,
                                color="#2ecc71",
                            ),
                            error_y_minus=(opt_df["평균"] - opt_df["적정 하한"]).astype(float),
                            mode="markers+lines",
                            marker=dict(size=10, color="#2ecc71", symbol="diamond"),
                            line=dict(color="#2ecc71", width=2),
                            name="평균",
                            hovertemplate="%{x}<br>평균: %{y:.2f}<extra></extra>",
                        )
                    )
                    fig_opt.update_layout(
                        title="환경 변수별 적정 구간 (30~70% 백분위 범위)",
                        xaxis_title="환경 변수",
                        yaxis_title="값",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        margin=dict(l=70, r=30, t=50, b=70),
                        height=380,
                    )
                    st.plotly_chart(fig_opt, width="stretch")

        # ── What-If 시뮬레이터 ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔬 What-If 시뮬레이터")
        st.caption(
            "슬라이더로 온도와 습도를 조절하면, 회귀식으로 예측되는 생육 결과값을 보여줍니다. "
            "이는 단순한 선형 근사이며 실제 생육은 여러 요인이 함께 작용합니다."
        )

        sim_cols = st.columns(2)
        with sim_cols[0]:
            sim_temp = st.slider(
                "온도 (°C)",
                min_value=-10.0,
                max_value=50.0,
                value=20.0,
                step=0.5,
            )
        with sim_cols[1]:
            sim_rh = st.slider(
                "습도 (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=1.0,
            )

        sim_available = (
            compare_col and compare_col in growth_df.columns
            and "temperature" in env_df.columns
            and "humidity" in env_df.columns
            and len(growth_df) >= 10
        )
        if sim_available:
            sim_data = pd.concat(
                [
                    env_df[["temperature", "humidity"]].dropna(),
                    growth_df[[compare_col]].dropna(),
                ],
                axis=1,
            ).dropna()

            if len(sim_data) >= 3:
                X_sim = sim_data[["temperature", "humidity"]].values.astype(float)
                y_sim = sim_data[compare_col].values.astype(float)

                try:
                    from sklearn.linear_model import LinearRegression

                    lr = LinearRegression()
                    lr.fit(X_sim, y_sim)

                    sim_point = np.array([[sim_temp, sim_rh]], dtype=float)
                    pred = float(lr.predict(sim_point)[0])
                    r2_sim = float(lr.score(X_sim, y_sim))

                    sim_name = {
                        "생중량": "생중량 (g)",
                        "잎 수": "잎 수 (장)",
                        "지상부 길이": "지상부 길이 (mm)",
                        "지하부 길이": "지하부 길이 (mm)",
                        "지상부 질량": "지상부 질량 (g)",
                        "지하부 질량": "지하부 질량 (g)",
                    }.get(compare_col, compare_col)

                    pred_cols = st.columns(3)
                    pred_cols[0].metric(
                        label="예측값",
                        value=f"{pred:.3f}",
                        delta=sim_name,
                    )
                    pred_cols[1].metric(
                        label="모델 설명력 (R²)",
                        value=f"{r2_sim:.3f}",
                    )
                    pred_cols[2].metric(
                        label="기준 온도",
                        value=f"{env_df['temperature'].mean():.1f} °C",
                    )

                    st.info(
                        f"🌡️ 온도 {sim_temp}°C · 습도 {sim_rh}% 조건에서의 "
                        f"예상 {sim_name}은 **{pred:.3f}**입니다. "
                        f"(선형 회귀 근사, R² = {r2_sim:.3f})"
                    )
                except ImportError:
                    st.warning(
                        "⚠️ scikit-learn이 설치되지 않아 시뮬레이션을 실행할 수 없어요. "
                        "requirements.txt에 scikit-learn을 추가하고 다시 실행하세요."
                    )
                except Exception as sim_e:
                    st.warning(f"⚠️ 시뮬레이션 계산 중 오류가 발생했어요: {sim_e}")
            else:
                st.info("⚠️ 시뮬레이션에 필요한 데이터가 충분하지 않아요.")
        else:
            st.info("⚠️ 시뮬레이션에 필요한 환경 열과 생육 열이 모두 없어요.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ── 푸터 ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"🏫 학교: {school}  ·  데이터 출처: CSV 업로드  ·  작성일: 2026-11-21"
)
