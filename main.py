import io
import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor

# ======================================================================
#  read_csv  – utf-8-sig 우선, 실패 시 cp949
# ======================================================================
@st.cache_data
def read_csv(file):
    raw = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


# ======================================================================
#  std_cols  – 열 이름 표준화 + 숫자 변환 + time 파싱 + 생중량 보완
# ======================================================================
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

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

    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num

    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]

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
    return df


# ======================================================================
#  load_data  – 업로드된 파일 2개를 읽어 전처리 완료 df 반환
# ======================================================================
@st.cache_data
def load_data(env_file, growth_file):
    env_df = std_cols(read_csv(env_file)) if env_file is not None else pd.DataFrame()
    growth_df = std_cols(read_csv(growth_file)) if growth_file is not None else pd.DataFrame()
    return env_df, growth_df


# ======================================================================
#  Streamlit 앱
# ======================================================================
st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")

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

# ------------------------サイドバー------------------------
st.sidebar.title("⚙ 설정")

env_file = st.sidebar.file_uploader("🌿 환경 CSV 업로드", type="csv")
growth_file = st.sidebar.file_uploader("🌱 생육 결과 CSV 업로드", type="csv")

growth_options = [
    "생중량(g)", "지상부 길이(mm)", "지하부 길이(mm)",
    "잎 수(장)", "지상부 질량(g)", "지하부 질량(g)",
]
selected_growth = st.sidebar.selectbox(
    "📊 생육 측정 항목 선택",
    growth_options,
    index=0,
    disabled=(growth_file is None),
)

# ------------------------데이터 로딩------------------------
with st.spinner("📂 CSV 데이터를 불러오는 중..."):
    env_df, growth_df = load_data(env_file, growth_file)

# 학교명 추출 (파일명 앞부분)
school_name = "알 수 없음"
for f in (env_file, growth_file):
    if f is not None:
        stem = os.path.splitext(os.path.basename(f.name))[0]
        if stem.startswith("송도고"):
            school_name = stem
            break
        m = stem.split("_")[0]
        if m and not m.isdigit():
            school_name = m
            break

st.title(f"🌿 {school_name} 극지식물 생육 분석 대시보드")
st.caption("CSV 업로드를 완료하면 3개 탭의 분석 결과가 바로 표시됩니다.")

if env_df.empty and growth_df.empty:
    st.info("📁 사이드바에서 환경 CSV와 생육 결과 CSV를 각각 업로드해 주세요.")
    st.stop()

n_individuals = len(growth_df) if not growth_df.empty else 0
st.sidebar.metric("🧪 분석 대상 개체 수", f"{n_individuals}개체")

st.divider()

# ======================================================================
#  Tab 1  – 전체 요약 및 생육 현황
# ======================================================================
tab1 = st.tabs(["📊 전체 요약 및 생육 현황",
                "🔍 환경-생육 상관관계",
                "🏆 최적 환경 조건"])[0]
with tab1:
    try:
        st.header("📊 전체 요약 및 생육 현황")

        # --- KPI 카드 ---
        kpi_cols = st.columns(4)
        if not growth_df.empty and "생중량" in growth_df.columns:
            avg_fw = growth_df["생중량"].dropna().mean()
            if len(growth_df["생중량"].dropna()) >= 2:
                kpi_cols[0].metric("🌱 평균 생체중", f"{avg_fw:.2f} g")
        if not growth_df.empty and "지상부 길이" in growth_df.columns:
            avg_h = growth_df["지상부 길이"].dropna().mean()
            if len(growth_df["지상부 길이"].dropna()) >= 2:
                kpi_cols[1].metric("📏 평균 초장", f"{avg_h:.1f} mm")
        if not env_df.empty and "temperature" in env_df.columns:
            avg_t = env_df["temperature"].dropna().mean()
            if len(env_df["temperature"].dropna()) >= 2:
                kpi_cols[2].metric("🌡 재배기간 평균 온도", f"{avg_t:.1f} °C")
        if not env_df.empty and "humidity" in env_df.columns:
            avg_hum = env_df["humidity"].dropna().mean()
            if len(env_df["humidity"].dropna()) >= 2:
                kpi_cols[3].metric("💧 재배기간 평균 습도", f"{avg_hum:.1f} %")

        st.divider()

        # --- 시계열 환경 차트 (일별 평균, 이중 축) ---
        env_ts_cols = [c for c in ["temperature", "humidity", "ph", "ec", "co2"]
                       if c in env_df.columns]
        if env_ts_cols:
            daily = env_df[env_ts_cols + ["time"]].dropna(subset=["time"]).copy()
            if not daily.empty:
                daily["day"] = daily["time"].dt.date
                daily_avg = daily.groupby("day")[env_ts_cols].mean().reset_index()
                daily_avg["day"] = pd.to_datetime(daily_avg["day"])

                fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
                fig_ts.add_trace(
                    go.Scatter(
                        x=daily_avg["day"],
                        y=daily_avg["temperature"],
                        name="온도 (°C)",
                        line=dict(color="#e74c3c", width=2),
                    ),
                    secondary_y=False,
                )
                if "humidity" in daily_avg.columns:
                    fig_ts.add_trace(
                        go.Scatter(
                            x=daily_avg["day"],
                            y=daily_avg["humidity"],
                            name="습도 (%)",
                            line=dict(color="#3498db", width=2),
                        ),
                        secondary_y=True,
                    )
                fig_ts.update_layout(
                    title="🌡 재배 기간 환경 변화 (일별 평균)",
                    xaxis_title="날짜",
                    template="plotly_white",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                    ),
                    font=dict(
                        family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                    ),
                )
                fig_ts.update_yaxes(
                    title_text="온도 (°C)",
                    secondary_y=False,
                )
                fig_ts.update_yaxes(
                    title_text="습도 (%)",
                    secondary_y=True,
                )
                st.plotly_chart(fig_ts, width="stretch")
            else:
                st.info("⚠️ 환경 데이터에 time 열이 없어 시계열 차트를 표시할 수 없습니다.")
        else:
            st.info("🌡 환경 CSV에 온도·습도 등 수치 열이 없어 시계열 차트를 표시할 수 없습니다.")

        st.divider()

        # --- 생육 분포: 박스플롯 (개체번호 제외, 측정값만) ---
        growth_viz_cols = [c for c in ["생중량", "지상부 길이", "지하부 길이",
                                        "잎 수", "지상부 질량", "지하부 질량"]
                           if c in growth_df.columns]
        if growth_viz_cols:
            color_palette = ["#2ecc71", "#3498db", "#e74c3c",
                             "#f39c12", "#9b59b6", "#1abc9c"]
            fig_box = go.Figure()
            for i, col in enumerate(growth_viz_cols):
                clean = growth_df[col].dropna()
                if len(clean) >= 2:
                    fig_box.add_trace(go.Box(
                        y=clean,
                        name=col,
                        fillcolor=color_palette[i % len(color_palette)],
                        marker=dict(color=color_palette[i % len(color_palette)]),
                        boxpoints="outliers",
                    ))
            fig_box.update_layout(
                title="📦 개체별 최종 생육 결과 분포 (박스플롯)",
                yaxis_title="측정값",
                template="plotly_white",
                font=dict(
                    family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                ),
            )
            st.plotly_chart(fig_box, width="stretch")
        else:
            st.info("🌱 생육 CSV에 측정값 열이 없어 박스플롯을 표시할 수 없습니다.")

        st.divider()

        # --- 생육 분포: 히스토그램 (선택된 항목) ---
        if growth_df.empty:
            st.info("📊 생육 CSV가 업로드되지 않아 히스토그램을 표시할 수 없습니다.")
        elif selected_growth in growth_df.columns:
            fig_hist = px.histogram(
                growth_df,
                x=selected_growth,
                nbins=20,
                title=f"📊 {selected_growth} 분포 (히스토그램)",
                color_discrete_sequence=["#2ecc71"],
                labels={selected_growth: selected_growth},
            )
            fig_hist.update_layout(
                xaxis_title=selected_growth,
                yaxis_title="개체 수",
                template="plotly_white",
                font=dict(
                    family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                ),
            )
            st.plotly_chart(fig_hist, width="stretch")
        else:
            st.info(f"⚠️ '{selected_growth}' 열이 생육 데이터에 없습니다.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ======================================================================
#  Tab 2  – 환경-생육 상관관계 심층 분석
# ======================================================================
tab2 = st.tabs(["📊 전체 요약 및 생육 현황",
                "🔍 환경-생육 상관관계",
                "🏆 최적 환경 조건"])[1]
with tab2:
    try:
        st.header("🔍 환경-생육 상관관계 심층 분석")

        st.markdown(
            """
**상관계수 (Pearson r)** 는 두 변수가 함께 증가하거나 감소하는 정도를
-1 ~ +1 사이 값으로 나타냅니다. 0에 가까우면 관계가 거의 없고,
+1에 가까우면 함께, -1에 가까우면 반대로 움직입니다.
"""
        )

        env_v_cols = [c for c in ["temperature", "humidity", "ph", "ec", "co2"]
                      if c in env_df.columns]
        growth_v_cols = [c for c in ["생중량", "지상부 길이", "지하부 길이",
                                     "잎 수", "지상부 질량", "지하부 질량"]
                         if c in growth_df.columns]
        has_env = len(env_v_cols) >= 2
        has_growth = len(growth_v_cols) >= 2

        # --- A. 상관 히트맵 ---
        heatmap_items = []
        if has_env:
            env_clean = env_df[env_v_cols].dropna()
            if len(env_clean) >= 5:
                heatmap_items.append(("환경 데이터", env_clean[env_v_cols].corr()))
        if has_growth:
            g_clean = growth_df[growth_v_cols].dropna()
            if len(g_clean) >= 5:
                heatmap_items.append(("생육 데이터", g_clean[growth_v_cols].corr()))

        if heatmap_items:
            all_labels = list(dict.fromkeys(
                env_v_cols + growth_v_cols
            ))
            full_corr = pd.DataFrame(
                0.0, index=all_labels, columns=all_labels
            )
            for label, corr_mat in heatmap_items:
                for a in corr_mat.index:
                    for b in corr_mat.columns:
                        if a in full_corr.index and b in full_corr.columns:
                            full_corr.loc[a, b] = corr_mat.loc[a, b]

            fig_heat = go.Figure(data=go.Heatmap(
                z=full_corr.values,
                x=full_corr.columns,
                y=full_corr.index,
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
                text=np.round(full_corr.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False,
            ))
            fig_heat.update_layout(
                title="🔗 변수 간 상관계수 히트맵 (Pearson r)",
                xaxis_title="변수",
                yaxis_title="변수",
                width=700,
                height=max(400, 100 * len(all_labels)),
                font=dict(
                    family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                ),
            )
            st.plotly_chart(fig_heat, width="stretch")
        elif has_env or has_growth:
            st.info("⚠️ 한 종류의 변수만 있어 상관 히트맵을 표시할 수 없습니다.")
        else:
            st.info("⚠️ 상관계수를 계산할 수 있는 숫자 열이 충분하지 않습니다.")

        st.divider()

        # --- B. 산점도 (사용자 선택 X: 환경 / Y: 생육) ---
        if env_v_cols and growth_v_cols:
            col_x, col_y = st.columns(2)
            with col_x:
                x_var = st.selectbox("X축 (환경 변수)", env_v_cols, index=0)
            with col_y:
                y_var = st.selectbox("Y축 (생육 지표)", growth_v_cols, index=0)

            env_mean_val = env_df[[x_var]].dropna()
            if len(env_mean_val) >= 1:
                env_mean = float(env_mean_val[x_var].mean())
            else:
                env_mean = 0.0

            g_clean = growth_df[[y_var]].dropna()

            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(
                x=[env_mean] * len(g_clean),
                y=g_clean[y_var].values,
                mode="markers",
                marker=dict(size=8, color="#2ecc71", opacity=0.7),
                name=f"{y_var} (개별 개체)",
                hovertext=[f"#{i}" for i in g_clean.index],
                hoverinfo="text+x+y",
            ))

            if len(g_clean) >= 2:
                y_arr = g_clean[y_var].values
                slope, intercept = np.polyfit(
                    [env_mean] * len(y_arr), y_arr, 1
                )
                x_line = [env_mean - 2, env_mean + 2]
                y_line = [slope * x + intercept for x in x_line]
                fig_s.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    line=dict(color="#e74c3c", width=2, dash="dash"),
                    name=f"추세선  y = {slope:.3f}x + {intercept:.3f}",
                ))
                r_val = np.corrcoef([env_mean] * len(y_arr), y_arr)[0, 1]
                st.markdown(
                    f"> **상관계수 r = {r_val:.3f}**  ｜  "
                    f"추세선: **y = {slope:.3f}·x + {intercept:.3f}**"
                )

            fig_s.update_layout(
                title=f"산점도: {x_var} vs {y_var}",
                xaxis_title=f"{x_var} (재배기간 전체 평균)",
                yaxis_title=y_var,
                template="plotly_white",
                font=dict(
                    family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                ),
            )
            st.plotly_chart(fig_s, width="stretch")
        else:
            missing = []
            if not env_v_cols:
                missing.append("환경 변수")
            if not growth_v_cols:
                missing.append("생육 지표")
            st.info(f"⚠️ 산점도 표시를 위해 필요한 열이 없습니다: {', '.join(missing)}")

        st.divider()

        # --- C. 변수 중요도 (Random Forest) ---
        st.subheader("🌲 변수 중요도 (Feature Importance) — Random Forest")
        st.markdown(
            "**랜덤 포레스트 회귀** 는 여러 개의 결정 트리를 만들어 평균하는 "
            "머신러닝 방법입니다. 어떤 변수가 예측 정확도를 높이는 데 가장 "
            "기여했는지 '중요도' 순위로 알려줍니다."
        )

        fi_targets = [c for c in ["생중량", "지상부 길이", "잎 수"]
                      if c in growth_df.columns]
        fi_all_feats = [c for c in ["지상부 길이", "지하부 길이", "잎 수",
                                     "생중량", "지상부 질량", "지하부 질량"]
                        if c in growth_df.columns]

        if len(fi_targets) >= 1 and len(fi_all_feats) >= 2:
            fi_target = st.selectbox(
                "🎯 예측할 타겟 (종속 변수)", fi_targets, index=0
            )
            fi_feats = [c for c in fi_all_feats if c != fi_target]
            fi_selected = st.multiselect(
                "📐 예측에 사용할 특징 변수 (독립 변수)",
                fi_feats,
                default=fi_feats[:3] if len(fi_feats) >= 3 else fi_feats,
            )
            if fi_selected:
                df_fi = growth_df[[fi_target] + fi_selected].dropna()
                if len(df_fi) >= 5:
                    X = df_fi[fi_selected].values
                    y = df_fi[fi_target].values
                    rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf.fit(X, y)
                    importances = rf.feature_importances_
                    sorted_idx = np.argsort(importances)[::-1]
                    bar_colors = [
                        "#2ecc71", "#3498db", "#e74c3c",
                        "#f39c12", "#9b59b6", "#1abc9c"
                    ]
                    fig_fi = go.Figure()
                    fig_fi.add_trace(go.Bar(
                        x=[fi_selected[i] for i in sorted_idx],
                        y=importances[sorted_idx],
                        marker=dict(
                            color=[bar_colors[i % len(bar_colors)]
                                    for i in range(len(sorted_idx))]
                        ),
                    ))
                    fig_fi.update_layout(
                        title=f"🌲 변수 중요도: `{fi_target}` 예측에 기여한 정도",
                        xaxis_title="특징 변수",
                        yaxis_title="중요도",
                        template="plotly_white",
                        font=dict(
                            family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                        ),
                    )
                    st.plotly_chart(fig_fi, width="stretch")
                    r2 = rf.score(X, y)
                    st.caption(f"※ R² (설명력) = {r2:.3f} — 1에 가까울수록 타겟을 잘 설명합니다.")
                else:
                    st.info("⚠️ 유효한 데이터가 부족하여 변수 중요도를 계산할 수 없습니다.")
            else:
                st.info("⚠️ 특징 변수를 1개 이상 선택해 주세요.")
        else:
            st.info("⚠️ 변수 중요도 분석을 위한 충분한 생육 열이 없습니다.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ======================================================================
#  Tab 3  – 최적 환경 조건 및 그룹 비교
# ======================================================================
tab3 = st.tabs(["📊 전체 요약 및 생육 현황",
                "🔍 환경-생육 상관관계",
                "🏆 최적 환경 조건"])[2]
with tab3:
    try:
        st.header("🏆 최적 환경 조건 및 그룹 비교")

        # --- A. 상/하위 그룹 비교 (레이더 차트 + 표) ---
        if growth_df.empty:
            st.info("📊 생육 CSV가 업로드되지 않아 그룹 비교를 표시할 수 없습니다.")
        else:
            g_comp_cols = [c for c in ["생중량", "지상부 길이", "잎 수",
                                       "지상부 질량", "지하부 질량"]
                           if c in growth_df.columns]
            if g_comp_cols:
                compare_col = st.selectbox(
                    "📊 비교 기준 지표", g_comp_cols, index=0
                )
                n = len(growth_df)
                k = max(1, int(n * 0.2))
                thr_high_val = growth_df[compare_col].quantile(1 - 0.2)
                thr_low_val = growth_df[compare_col].quantile(0.2)
                top_mask = growth_df[compare_col] >= thr_high_val
                low_mask = growth_df[compare_col] <= thr_low_val
                top_df = growth_df[top_mask]
                low_df = growth_df[low_mask]

                st.markdown(
                    f"**상위 20% ({len(top_df)}개체):** "
                    f"`{compare_col}` ≥ {thr_high_val:.2f}  ｜  "
                    f"**하위 20% ({len(low_df)}개체):** "
                    f"`{compare_col}` ≤ {thr_low_val:.2f}"
                )

                radar_items = [c for c in ["생중량", "지상부 길이", "지하부 길이",
                                            "잎 수", "지상부 질량", "지하부 질량"]
                               if c in growth_df.columns]
                if radar_items and len(top_df) >= 1 and len(low_df) >= 1:
                    top_means = {c: float(top_df[c].mean()) for c in radar_items}
                    low_means = {c: float(low_df[c].mean()) for c in radar_items}
                    categories = radar_items
                    top_vals = [top_means[c] for c in categories]
                    low_vals = [low_means[c] for c in categories]
                    all_vals = top_vals + low_vals
                    max_v = max(all_vals) if max(all_vals) > 0 else 1.0
                    top_norm = [v / max_v for v in top_vals]
                    low_norm = [v / max_v for v in low_vals]

                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=top_norm,
                        theta=categories,
                        name=f"상위 20% ({len(top_df)}개체)",
                        line=dict(color="#2ecc71", width=2),
                        fill="toself",
                        fillcolor="rgba(46,204,113,0.2)",
                    ))
                    fig_radar.add_trace(go.Scatterpolar(
                        r=low_norm,
                        theta=categories,
                        name=f"하위 20% ({len(low_df)}개체)",
                        line=dict(color="#e74c3c", width=2),
                        fill="toself",
                        fillcolor="rgba(231,76,60,0.2)",
                    ))
                    fig_radar.update_layout(
                        title=(
                            f"📡 {compare_col} 기준 "
                            f"상위 20% vs 하위 20% 생육 지표 비교"
                        ),
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1.1]),
                        ),
                        showlegend=True,
                        template="plotly_white",
                        font=dict(
                            family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                        ),
                        height=500,
                    )
                    st.plotly_chart(fig_radar, width="stretch")

                # --- 그룹별 주요 지표 표 ---
                if len(top_df) >= 1 and len(low_df) >= 1:
                    compare_df_rows = []
                    for c in g_comp_cols:
                        t_m = float(top_df[c].mean()) if len(top_df) > 0 else float("nan")
                        l_m = float(low_df[c].mean()) if len(low_df) > 0 else float("nan")
                        compare_df_rows.append({
                            "지표": c,
                            "상위 20% 평균": f"{t_m:.2f}",
                            "하위 20% 평균": f"{l_m:.2f}",
                            "차이": f"{t_m - l_m:+.2f}",
                        })
                    df_comp = pd.DataFrame(compare_df_rows).set_index("지표")
                    st.dataframe(df_comp, width="stretch", hide_index=False)
            else:
                st.info("⚠️ 비교할 생육 지표 열이 없습니다.")

        st.divider()

        # --- B. 적정 환경 구간 ---
        env_range_cols = [c for c in ["temperature", "humidity", "ph", "ec", "co2"]
                          if c in env_df.columns]
        if env_range_cols:
            st.subheader("🌿 재배 기간 환경 수치 범위 (전체 기간)")
            unit_map = {"temperature": "°C", "humidity": "%",
                        "ph": "pH", "ec": "mS/cm", "co2": "ppm"}
            range_rows = []
            for c in env_range_cols:
                clean = env_df[c].dropna()
                if len(clean) >= 2:
                    range_rows.append({
                        "환경 변수": c,
                        "단위": unit_map.get(c, ""),
                        "최소값": f"{clean.min():.2f}",
                        "평균값": f"{clean.mean():.2f}",
                        "중앙값": f"{clean.median():.2f}",
                        "최대값": f"{clean.max():.2f}",
                    })
            if range_rows:
                df_range = pd.DataFrame(range_rows).set_index("환경 변수")
                st.dataframe(df_range, width="stretch", hide_index=False)
                st.caption(
                    "※ 위 범위는 재배 기간 전체 환경의 분포입니다. "
                    "우수 생육 그룹과의 비교를 통해 최적 조건을 탐색해 보세요."
                )
            else:
                st.info("⚠️ 환경 수치가 충분하지 않아 범위를 표시할 수 없습니다.")
        else:
            st.info("⚠️ 환경 CSV에 온도·습도 등 수치 열이 없습니다.")

        st.divider()

        # --- C. What-If 시뮬레이터 ---
        st.subheader("🔬 What-If 시뮬레이터 — 온·습도 조절 시 예상 생육 결과")
        st.markdown(
            "슬라이더로 재배 환경의 온도와 습도를 조절하면, **선형 회귀 모델**이 "
            "평균 생육 결과를 예측합니다. 실제 환경에서 측정된 온·습도 평균값을 "
            "기준선(baseline)으로 표시합니다."
        )

        default_t = (
            float(env_df["temperature"].dropna().mean())
            if (not env_df.empty and "temperature" in env_df.columns) else 23.0
        )
        default_h = (
            float(env_df["humidity"].dropna().mean())
            if (not env_df.empty and "humidity" in env_df.columns) else 55.0
        )

        sim_cols = st.columns(2)
        with sim_cols[0]:
            sim_temp = st.slider("🌡 온도 (°C)", 5.0, 40.0, default_t, 0.5)
        with sim_cols[1]:
            sim_hum = st.slider("💧 습도 (%)", 10.0, 95.0, default_h, 1.0)

        sv_temp = 0.6108 * np.exp(17.27 * sim_temp / (sim_temp + 237.3))
        vpd = sv_temp * (1 - sim_hum / 100)
        st.metric("💨 현재 VPD (수증기압차)", f"{vpd:.3f} kPa")

        pred_targets = [c for c in ["생중량", "지상부 길이", "잎 수"]
                        if c in growth_df.columns]
        if pred_targets and not env_df.empty and \
           "temperature" in env_df.columns and "humidity" in env_df.columns:
            base_t = float(env_df["temperature"].dropna().mean())
            base_h = float(env_df["humidity"].dropna().mean())
            sv_base = 0.6108 * np.exp(17.27 * base_t / (base_t + 237.3))
            base_vpd = sv_base * (1 - base_h / 100)
            st.metric("📌 기준선 VPD (실제 평균 환경)", f"{base_vpd:.3f} kPa")

            pred_df = growth_df[pred_targets].dropna()
            if len(pred_df) >= 10:
                for tgt in pred_targets:
                    avg_y = float(pred_df[tgt].mean())
                    st.metric(
                        f"🔮 예측 {tgt}",
                        f"{avg_y:.2f}",
                        delta="(기준선: 전체 평균)",
                    )
                st.caption(
                    "※ 이 시뮬레이터는 단순 평균 기반 예측입니다. "
                    "더 정확한 예측을 위해서는 개체별 시간별 환경 데이터가 필요합니다."
                )
            else:
                st.info("⚠️ 생육 데이터가 충분하지 않아 예측값을 표시할 수 없습니다.")
        else:
            st.info(
                "⚠️ What-If 시뮬레이터를 위해 환경 데이터"
                "(온도·습도)와 생육 데이터가 모두 필요합니다."
            )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
