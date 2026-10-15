
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# 1. 페이지 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="AI4I 설비 상태 분석",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 제조 설비 상태 분석 대시보드")
st.caption(
    "AI4I 2020 Predictive Maintenance Dataset | "
    "설비 운전 조건 및 고장 현황 탐색"
)


# --------------------------------------------------
# 2. 데이터 불러오기 및 파생변수 생성
# --------------------------------------------------
DATA_PATH = Path("data/ai4i2020.csv")


@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)

    # 필수 컬럼 확인
    required_cols = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure",
    ]

    missing_cols = [
        col for col in required_cols if col not in data.columns
    ]

    if missing_cols:
        raise ValueError(
            f"필수 컬럼이 없습니다: {missing_cols}"
        )

    # 파생변수
    data["Air temperature [C]"] = (
        data["Air temperature [K]"] - 273.15
    )

    data["Process temperature [C]"] = (
        data["Process temperature [K]"] - 273.15
    )

    data["T_distance"] = (
        data["Process temperature [K]"]
        - data["Air temperature [K]"]
    )

    data["Power [W]"] = (
        data["Torque [Nm]"]
        * data["Rotational speed [rpm]"]
        * 2 * np.pi / 60
    )

    data["설비 상태"] = data["Machine failure"].map(
        {0: "정상", 1: "고장"}
    )

    return data


if not DATA_PATH.exists():
    st.error(
        f"데이터 파일을 찾을 수 없습니다: {DATA_PATH.resolve()}"
    )
    st.info(
        "프로젝트 폴더에 data 폴더를 만들고 "
        "CSV 파일을 넣어주세요."
    )
    st.stop()

try:
    df = load_data(str(DATA_PATH))
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 3. 공통 필터
# --------------------------------------------------
st.sidebar.header("분석 조건")

page = st.sidebar.radio(
    "대시보드 페이지",
    [
        "전체 설비 현황",
        "온도 분석",
        "회전·부하 분석",
    ],
)

types = sorted(df["Type"].dropna().unique().tolist())
type_options = ["전체"] + types

selected_type = st.sidebar.selectbox(
    "제품 유형 (Type)",
    type_options,
)

if selected_type == "전체":
    filtered_df = df.copy()
else:
    filtered_df = df[df["Type"] == selected_type].copy()

st.sidebar.caption(
    f"선택 데이터: {len(filtered_df):,}건"
)


# --------------------------------------------------
# 4. 공통 계산 함수
# --------------------------------------------------
def failure_rate(data):
    """Machine failure가 0 또는 1로 기록된 행만 사용."""
    valid = data["Machine failure"].dropna()
    if len(valid) == 0:
        return np.nan

    return valid.eq(1).mean() * 100


def show_kpi(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def style_chart(fig):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="",
    )
    return fig


STATUS_COLORS = {"정상": "#66CCFF", "고장": "#E53935"}


# --------------------------------------------------
# 5. 페이지 1: 전체 설비 현황
# --------------------------------------------------
if page == "전체 설비 현황":
    st.header("전체 설비 현황")

    total_count = len(filtered_df)
    failures = filtered_df["Machine failure"].eq(1).sum()
    rate = failure_rate(filtered_df)

    avg_wear = filtered_df["Tool wear [min]"].mean()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        show_kpi("전체 관측 건수", f"{total_count:,}")

    with c2:
        show_kpi("고장 관측 건수", f"{failures:,}")

    with c3:
        show_kpi(
            "고장률",
            f"{rate:.2f}%" if pd.notna(rate) else "N/A",
            "선택한 데이터 중 고장으로 기록된 비율",
        )

    with c4:
        show_kpi(
            "평균 공구 마모 시간",
            f"{avg_wear:.1f} min"
            if pd.notna(avg_wear) else "N/A",
        )

    st.divider()

    left, right = st.columns(2)

    # 제품 유형별 관측 건수
    with left:
        st.subheader("제품 유형별 관측 건수")

        count_df = (
            filtered_df.groupby("Type", as_index=False)
            .size()
            .rename(columns={"size": "관측 건수"})
        )

        fig = px.bar(
            count_df,
            x="Type",
            y="관측 건수",
            text="관측 건수",
            title="Type별 데이터 규모",
        )
        st.plotly_chart(
            style_chart(fig),
            use_container_width=True,
        )

    # 유형별 고장률
    with right:
        st.subheader("제품 유형별 고장률")

        rate_df = (
            filtered_df.groupby("Type")
            .agg(
                전체=("Machine failure", "count"),
                고장=("Machine failure", lambda x: x.eq(1).sum()),
            )
            .reset_index()
        )

        rate_df["고장률 (%)"] = (
            rate_df["고장"] / rate_df["전체"] * 100
        )

        fig = px.bar(
            rate_df,
            x="Type",
            y="고장률 (%)",
            text=rate_df["고장률 (%)"].map(
                lambda x: f"{x:.2f}%"
            ),
            title="Type별 고장률 비교",
        )
        st.plotly_chart(
            style_chart(fig),
            use_container_width=True,
        )

    st.subheader("정상·고장 관측 비율")

    status_df = (
        filtered_df["설비 상태"]
        .value_counts()
        .rename_axis("설비 상태")
        .reset_index(name="관측 건수")
    )

    fig = px.pie(
        status_df,
        names="설비 상태",
        values="관측 건수",
        hole=0.45,
        color="설비 상태",
        color_discrete_map=STATUS_COLORS,
        title="정상·고장 구성",
    )
    st.plotly_chart(
        style_chart(fig),
        use_container_width=True,
    )


# --------------------------------------------------
# 6. 페이지 2: 온도 분석
# --------------------------------------------------
elif page == "온도 분석":
    st.header("온도 분석")
    st.caption(
        "공기 온도, 공정 온도 및 두 온도의 차이를 비교합니다."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        show_kpi(
            "평균 공기 온도",
            f"{filtered_df['Air temperature [C]'].mean():.2f} °C",
        )

    with c2:
        show_kpi(
            "평균 공정 온도",
            f"{filtered_df['Process temperature [C]'].mean():.2f} °C",
        )

    with c3:
        show_kpi(
            "평균 온도 차이",
            f"{filtered_df['T_distance'].mean():.2f} °C",
        )

    st.divider()

    left, right = st.columns(2)

    # 온도 분포
    with left:
        st.subheader("공기 온도 분포")

        fig = px.histogram(
            filtered_df,
            x="Air temperature [C]",
            color="설비 상태",
            barmode="overlay",
            opacity=0.65,
            nbins=30,
            labels={
                "Air temperature [C]": "공기 온도 (°C)",
                "설비 상태": "설비 상태",
            },
            color_discrete_map=STATUS_COLORS,
            title="공기 온도 분포: 정상·고장 비교",
        )
        st.plotly_chart(
            style_chart(fig),
            use_container_width=True,
        )

    with right:
        st.subheader("공정 온도 분포")

        fig = px.histogram(
            filtered_df,
            x="Process temperature [C]",
            color="설비 상태",
            barmode="overlay",
            opacity=0.65,
            nbins=30,
            labels={
                "Process temperature [C]": "공정 온도 (°C)",
                "설비 상태": "설비 상태",
            },
            color_discrete_map=STATUS_COLORS,
            title="공정 온도 분포: 정상·고장 비교",
        )
        st.plotly_chart(
            style_chart(fig),
            use_container_width=True,
        )

    st.subheader("온도 차이와 고장 여부")

    fig = px.box(
        filtered_df,
            x="설비 상태",
        y="T_distance",
        points="outliers",
        labels={
                "설비 상태": "설비 상태",
            "T_distance": "공정 온도 - 공기 온도 (°C)",
        },
        title="고장 여부별 온도 차이 분포",
    )
    st.plotly_chart(
        style_chart(fig),
        use_container_width=True,
    )


# --------------------------------------------------
# 7. 페이지 3: 회전·부하 분석
# --------------------------------------------------
elif page == "회전·부하 분석":
    st.header("회전·부하 분석")
    st.caption(
        "회전속도, 토크 및 계산된 기계적 동력을 탐색합니다."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        show_kpi(
            "평균 회전속도",
            f"{filtered_df['Rotational speed [rpm]'].mean():,.1f} rpm",
        )

    with c2:
        show_kpi(
            "평균 토크",
            f"{filtered_df['Torque [Nm]'].mean():.2f} Nm",
        )

    with c3:
        show_kpi(
            "평균 추정 기계 동력",
            f"{filtered_df['Power [W]'].mean():,.1f} W",
        )

    st.divider()

    left, right = st.columns(2)

    # 회전속도와 토크
    with left:
        st.subheader("회전속도와 토크의 관계")

        fig = px.scatter(
            filtered_df,
            x="Rotational speed [rpm]",
            y="Torque [Nm]",
            color="설비 상태",
            opacity=0.65,
            labels={
                "Rotational speed [rpm]": "회전속도 (rpm)",
                "Torque [Nm]": "토크 (Nm)",
                "설비 상태": "설비 상태",
            },
            color_discrete_map=STATUS_COLORS,
            title="회전속도 vs 토크",
        )
        st.plotly_chart(
            style_chart(fig),
            use_container_width=True,
        )

    # 공구 마모와 추정 기계 동력
    with right:
        st.subheader("공구 마모와 추정 기계 동력")

        fig = px.scatter(
            filtered_df,
            x="Tool wear [min]",
            y="Power [W]",
            color="설비 상태",
            opacity=0.65,
            labels={
                "Tool wear [min]": "공구 마모 시간 (min)",
                "Power [W]": "추정 기계 동력 (W)",
                "설비 상태": "설비 상태",
            },
            color_discrete_map=STATUS_COLORS,
            title="공구 마모 vs 추정 기계 동력",
        )
        st.plotly_chart(
            style_chart(fig),
            use_container_width=True,
        )

    st.subheader("운전 변수 요약")

    summary = (
        filtered_df.groupby("Machine failure")
        .agg(
            관측_건수=("Machine failure", "size"),
            평균_회전속도=("Rotational speed [rpm]", "mean"),
            평균_토크=("Torque [Nm]", "mean"),
            평균_공구마모=("Tool wear [min]", "mean"),
            평균_동력=("Power [W]", "mean"),
        )
        .rename(index={0: "정상", 1: "고장"})
        .round(2)
    )

    st.dataframe(summary, use_container_width=True)


# --------------------------------------------------
# 8. 데이터 미리보기
# --------------------------------------------------
with st.expander("현재 필터에 해당하는 데이터 보기"):
    preview_df = filtered_df.drop(columns=["Machine failure"])
    st.dataframe(
        preview_df.head(100),
        use_container_width=True,
    )

st.caption(
    "주의: 본 데이터는 합성 데이터이며, 관찰된 관계가 "
    "실제 제조 현장의 인과관계를 증명하지는 않습니다."
)
