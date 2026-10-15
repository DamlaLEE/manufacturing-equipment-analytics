from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="AI4I 설비 상태 분석", page_icon="🏭", layout="wide")
st.title("🏭 제조 설비 상태 분석 대시보드")
st.caption("AI4I 2020 Predictive Maintenance Dataset | 설비 운전 조건 및 고장 현황 탐색")

STATUS_COLORS = {"정상": "#66CCFF", "고장": "#E53935"}
DATA_PATH = Path("data/ai4i2020.csv")


@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    required = [
        "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
        "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF",
    ]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    data["Air temperature [C]"] = data["Air temperature [K]"] - 273.15
    data["Process temperature [C]"] = data["Process temperature [K]"] - 273.15
    data["T_distance"] = data["Process temperature [K]"] - data["Air temperature [K]"]
    data["Power [W]"] = (
        data["Torque [Nm]"] * data["Rotational speed [rpm]"] * 2 * np.pi / 60
    )
    data["설비 상태"] = data["Machine failure"].map({0: "정상", 1: "고장"})

    # 탐색용 구간: 실제 현장의 공식 관리 기준이 아님
    data["Tool wear band"] = pd.cut(
        data["Tool wear [min]"],
        bins=[-np.inf, 60, 120, 180, np.inf],
        labels=["60분 이하", "60~120분", "120~180분", "180분 초과"],
    )
    data["Temperature difference band"] = pd.cut(
        data["T_distance"],
        bins=[-np.inf, 8, 10, 12, np.inf],
        labels=["8°C 미만", "8~10°C", "10~12°C", "12°C 이상"],
        right=False,
    )
    return data


if not DATA_PATH.exists():
    st.error(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH.resolve()}")
    st.info("app.py와 같은 폴더 안에 data/ai4i2020.csv 파일이 있는지 확인하세요.")
    st.stop()

try:
    df = load_data(str(DATA_PATH))
except Exception as exc:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {exc}")
    st.stop()

st.sidebar.header("분석 조건")
page = st.sidebar.radio(
    "대시보드 페이지",
    ["전체 설비 현황", "온도 분석", "회전·부하 분석", "고장 유형 분석"],
)
type_options = ["전체"] + sorted(df["Type"].dropna().unique().tolist())
selected_type = st.sidebar.selectbox("제품 유형 (Type)", type_options)
filtered_df = df.copy() if selected_type == "전체" else df[df["Type"] == selected_type].copy()
st.sidebar.caption(f"선택 데이터: {len(filtered_df):,}건")


def style_chart(fig):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="",
    )
    return fig


def failure_rate(data):
    values = data["Machine failure"].dropna()
    return values.eq(1).mean() * 100 if not values.empty else np.nan


def make_rate_summary(data, group_col):
    result = (
        data.groupby(group_col, observed=False)
        .agg(
            관측건수=("Machine failure", "count"),
            고장건수=("Machine failure", lambda x: x.eq(1).sum()),
        )
        .reset_index()
    )
    result["고장률 (%)"] = (
        result["고장건수"] / result["관측건수"].replace(0, np.nan) * 100
    )
    return result


if page == "전체 설비 현황":
    st.header("전체 설비 현황")
    st.caption("전체 규모, 고장 현황, 공구 마모 특성을 빠르게 확인합니다.")
    total = len(filtered_df)
    failures = int(filtered_df["Machine failure"].eq(1).sum())
    rate = failure_rate(filtered_df)
    avg_wear = filtered_df["Tool wear [min]"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 관측 건수", f"{total:,}")
    c2.metric("고장 관측 건수", f"{failures:,}")
    c3.metric("고장률", f"{rate:.2f}%" if pd.notna(rate) else "N/A")
    c4.metric("평균 공구 마모 시간", f"{avg_wear:.1f} min" if pd.notna(avg_wear) else "N/A")
    st.divider()

    left, right = st.columns(2)
    with left:
        counts = filtered_df.groupby("Type", as_index=False).size().rename(columns={"size": "관측 건수"})
        fig = px.bar(counts, x="Type", y="관측 건수", text="관측 건수", title="Type별 데이터 규모")
        st.plotly_chart(style_chart(fig), use_container_width=True)
    with right:
        rates = filtered_df.groupby("Type").agg(
            전체=("Machine failure", "count"),
            고장=("Machine failure", lambda x: x.eq(1).sum()),
        ).reset_index()
        rates["고장률 (%)"] = rates["고장"] / rates["전체"].replace(0, np.nan) * 100
        fig = px.bar(rates, x="Type", y="고장률 (%)", text=rates["고장률 (%)"].map(lambda x: f"{x:.2f}%"), title="Type별 고장률 비교")
        st.plotly_chart(style_chart(fig), use_container_width=True)

    status = filtered_df["설비 상태"].value_counts().rename_axis("설비 상태").reset_index(name="관측 건수")
    fig = px.pie(status, names="설비 상태", values="관측 건수", hole=0.45,
                 color="설비 상태", color_discrete_map=STATUS_COLORS, title="정상·고장 구성")
    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.divider()
    st.subheader("공구 마모 구간별 고장률")
    st.caption("구간은 탐색용이며 실제 공구 교체 기준을 뜻하지 않습니다.")
    wear = make_rate_summary(filtered_df, "Tool wear band")
    left, right = st.columns(2)
    with left:
        fig = px.bar(wear, x="Tool wear band", y="고장률 (%)",
                     text=wear["고장률 (%)"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"),
                     title="마모 구간별 고장률")
        st.plotly_chart(style_chart(fig), use_container_width=True)
    with right:
        fig = px.bar(wear, x="Tool wear band", y="관측건수", text="관측건수", title="마모 구간별 관측 건수")
        st.plotly_chart(style_chart(fig), use_container_width=True)
    st.dataframe(wear, use_container_width=True)

elif page == "온도 분석":
    st.header("온도 분석")
    st.caption("공기 온도, 공정 온도 및 두 온도의 차이를 비교합니다.")
    c1, c2, c3 = st.columns(3)
    c1.metric("평균 공기 온도", f"{filtered_df['Air temperature [C]'].mean():.2f} °C")
    c2.metric("평균 공정 온도", f"{filtered_df['Process temperature [C]'].mean():.2f} °C")
    c3.metric("평균 온도 차이", f"{filtered_df['T_distance'].mean():.2f} °C")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(filtered_df, x="Air temperature [C]", color="설비 상태",
                           barmode="overlay", opacity=0.65, nbins=30,
                           color_discrete_map=STATUS_COLORS, title="공기 온도 분포: 정상·고장 비교",
                           labels={"Air temperature [C]": "공기 온도 (°C)"})
        st.plotly_chart(style_chart(fig), use_container_width=True)
    with right:
        fig = px.histogram(filtered_df, x="Process temperature [C]", color="설비 상태",
                           barmode="overlay", opacity=0.65, nbins=30,
                           color_discrete_map=STATUS_COLORS, title="공정 온도 분포: 정상·고장 비교",
                           labels={"Process temperature [C]": "공정 온도 (°C)"})
        st.plotly_chart(style_chart(fig), use_container_width=True)

    fig = px.box(filtered_df, x="설비 상태", y="T_distance", color="설비 상태",
                 points="outliers", color_discrete_map=STATUS_COLORS,
                 title="정상·고장 그룹의 온도 차이 비교",
                 labels={"T_distance": "공정 온도 - 공기 온도 (°C)"})
    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.divider()
    st.subheader("온도 차이 구간별 고장률")
    st.caption("구간은 탐색용이며 공정의 공식 관리 한계값이 아닙니다.")
    temp = make_rate_summary(filtered_df, "Temperature difference band")
    fig = px.bar(temp, x="Temperature difference band", y="고장률 (%)",
                 text=temp["고장률 (%)"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"),
                 title="온도 차이 구간별 고장률")
    st.plotly_chart(style_chart(fig), use_container_width=True)
    st.dataframe(temp, use_container_width=True)

elif page == "회전·부하 분석":
    st.header("회전·부하 분석")
    st.caption("동력은 토크와 회전속도로 계산한 추정 기계 동력이며 실제 전력 소비량이 아닙니다.")
    c1, c2, c3 = st.columns(3)
    c1.metric("평균 회전속도", f"{filtered_df['Rotational speed [rpm]'].mean():,.1f} rpm")
    c2.metric("평균 토크", f"{filtered_df['Torque [Nm]'].mean():.2f} Nm")
    c3.metric("평균 추정 기계 동력", f"{filtered_df['Power [W]'].mean():,.1f} W")

    left, right = st.columns(2)
    with left:
        fig = px.scatter(filtered_df, x="Rotational speed [rpm]", y="Torque [Nm]",
                         color="설비 상태", opacity=0.65, color_discrete_map=STATUS_COLORS,
                         title="회전속도 vs 토크",
                         labels={"Rotational speed [rpm]": "회전속도 (rpm)", "Torque [Nm]": "토크 (N·m)"})
        st.plotly_chart(style_chart(fig), use_container_width=True)
    with right:
        fig = px.scatter(filtered_df, x="Tool wear [min]", y="Power [W]",
                         color="설비 상태", opacity=0.65, color_discrete_map=STATUS_COLORS,
                         title="공구 마모 vs 추정 기계 동력",
                         labels={"Tool wear [min]": "공구 마모 시간 (min)", "Power [W]": "추정 기계 동력 (W)"})
        st.plotly_chart(style_chart(fig), use_container_width=True)

    st.subheader("정상·고장 그룹별 운전 변수 요약")
    summary = filtered_df.groupby("설비 상태", dropna=False).agg(
        관측_건수=("Machine failure", "size"),
        평균_회전속도=("Rotational speed [rpm]", "mean"),
        평균_토크=("Torque [Nm]", "mean"),
        평균_공구마모=("Tool wear [min]", "mean"),
        평균_동력=("Power [W]", "mean"),
    ).round(2)
    st.dataframe(summary, use_container_width=True)

elif page == "고장 유형 분석":
    st.header("고장 유형 분석")
    st.caption("유형별 플래그를 비교해 추가 조사가 필요한 유형을 탐색합니다. 한 행에 여러 플래그가 있을 수 있습니다.")
    failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    labels = {
        "TWF": "TWF · 공구 마모 관련",
        "HDF": "HDF · 방열/온도 관련",
        "PWF": "PWF · 동력 관련",
        "OSF": "OSF · 과부하 관련",
        "RNF": "RNF · 무작위 고장 유형",
    }
    failure_only = filtered_df[filtered_df["Machine failure"].eq(1)]
    counts = failure_only[failure_cols].sum().sort_values(ascending=False)
    pareto = counts.rename_axis("코드").reset_index(name="유형별 플래그 건수")
    pareto["고장 유형"] = pareto["코드"].map(labels)
    total_flags = pareto["유형별 플래그 건수"].sum()
    pareto["누적 비중 (%)"] = (
        pareto["유형별 플래그 건수"].cumsum() / total_flags * 100
        if total_flags > 0 else 0.0
    )
    c1, c2 = st.columns(2)
    c1.metric("고장 관측 건수", f"{len(failure_only):,}")
    c2.metric("고장 유형 플래그 합계", f"{int(total_flags):,}")

    fig = px.bar(pareto, x="고장 유형", y="유형별 플래그 건수",
                 text="유형별 플래그 건수", title="고장 유형별 플래그 건수")
    st.plotly_chart(style_chart(fig), use_container_width=True)
    fig = px.line(pareto, x="고장 유형", y="누적 비중 (%)", markers=True,
                  text=pareto["누적 비중 (%)"].map(lambda x: f"{x:.1f}%"),
                  title="고장 유형 플래그 누적 비중")
    fig.update_yaxes(range=[0, 105])
    st.plotly_chart(style_chart(fig), use_container_width=True)
    st.dataframe(pareto, use_container_width=True)
    st.info("유형별 플래그는 중복될 수 있습니다. 누적 비중은 전체 고장 건수가 아니라 모든 유형 플래그 합계 기준입니다.")

with st.expander("현재 필터에 해당하는 데이터 보기"):
    st.dataframe(filtered_df.head(100), use_container_width=True)
    st.download_button(
        "현재 필터 데이터 CSV 다운로드",
        data=filtered_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="ai4i_filtered_data.csv",
        mime="text/csv",
    )

st.caption("AI4I는 합성 데이터입니다. 관찰된 관계는 인과관계를 증명하지 않으며, 구간별 고장률은 실제 현장 관리 기준을 대신하지 않습니다.")
