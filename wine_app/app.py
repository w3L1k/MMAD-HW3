"""Интерактивная оценка качества вина и поведения классификатора."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model.joblib"
TEST_PATH = APP_DIR / "test.csv"
CONFIG_PATH = APP_DIR / "config.json"


# Модель — общий изменяемый объект на сервере, поэтому cache_resource.
# Таблица и массив вероятностей — данные: cache_data отдаёт копии и изолирует сессии.
# Без декоратора joblib.load выполнялся бы при каждом движении ползунка.
@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


@st.cache_data
def load_test_data(path: Path):
    return pd.read_csv(path)


@st.cache_data
def test_proba(_model, X: pd.DataFrame):
    """Порог не входит в аргументы: при его изменении предсказания не пересчитываются."""
    return _model.predict_proba(X)[:, 1]


def label_prediction(probability: float, threshold: float) -> str:
    return "Хорошее вино" if probability >= threshold else "Не рекомендовать"


st.set_page_config(page_title="Сомелье: качество вина", page_icon="🍷", layout="wide")
st.title("🍷 Оценка качества вина")
st.caption("Вероятность экспертной оценки не ниже 7 баллов по лабораторным признакам")

try:
    model = load_model(MODEL_PATH)
    test_df = load_test_data(TEST_PATH)
    default_threshold = float(json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["threshold"])
except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
    st.error(f"Не удалось загрузить файлы приложения: {exc}")
    st.stop()

feature_names = list(model.feature_names_in_)
if "target" not in test_df or not set(feature_names).issubset(test_df.columns):
    st.error("В test.csv не хватает target или признаков обученной модели.")
    st.stop()

X_test = test_df.loc[:, feature_names]
y_test = test_df["target"].astype(int)
test_probabilities = test_proba(model, X_test)

st.sidebar.header("Настройки рекомендации")
threshold = st.sidebar.slider(
    "Порог вероятности",
    min_value=0.0,
    max_value=1.0,
    value=default_threshold,
    step=0.001,
    format="%.3f",
    help="Рекомендовать вино, если вероятность хорошего качества не ниже порога.",
)
st.sidebar.caption(f"Порог из config.json: {default_threshold:.3f}")

single_tab, metrics_tab, batch_tab, weights_tab = st.tabs(
    ["Оценка одного вина", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"]
)

with single_tab:
    st.subheader("Лабораторные показатели")
    st.write("Укажите характеристики одного вина. Диапазоны взяты из тестового набора.")
    values = {}
    columns = st.columns(3)
    for i, feature in enumerate(feature_names):
        if feature == "is_red":
            values[feature] = int(
                columns[i % 3].toggle("Красное вино", value=bool(X_test[feature].median()))
            )
        else:
            series = X_test[feature]
            low, high, median = map(float, (series.min(), series.max(), series.median()))
            step = min(1.0, 10.0 ** np.floor(np.log10((high - low) / 100)))
            values[feature] = columns[i % 3].number_input(
                feature,
                min_value=low,
                max_value=high,
                value=median,
                step=step,
                format="%.5f",
                key=f"single_{feature}",
            )

    one_wine = pd.DataFrame([values], columns=feature_names)
    probability = float(model.predict_proba(one_wine)[0, 1])
    col_probability, col_decision = st.columns(2)
    col_probability.metric("Вероятность хорошего вина", f"{probability:.1%}")
    col_decision.metric("Решение", label_prediction(probability, threshold))

with metrics_tab:
    st.subheader("Качество на отложенной тестовой выборке")
    y_pred = (test_probabilities >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    precision = metrics.precision_score(y_test, y_pred, zero_division=0)
    recall = metrics.recall_score(y_test, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_test, y_pred, zero_division=0)

    p_col, r_col, f_col = st.columns(3)
    p_col.metric("Precision", f"{precision:.3f}")
    r_col.metric("Recall", f"{recall:.3f}")
    f_col.metric("F₁", f"{f1:.3f}")

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics.ConfusionMatrixDisplay(
        confusion_matrix=np.array([[tn, fp], [fn, tp]]),
        display_labels=["Не хорошее", "Хорошее"],
    ).plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Матрица ошибок")

    fpr, tpr, _ = metrics.roc_curve(y_test, test_probabilities)
    axes[1].plot(fpr, tpr, label=f"ROC-AUC = {metrics.auc(fpr, tpr):.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].scatter(fp / (fp + tn), recall, color="crimson", s=65,
                    zorder=3, label=f"Порог {threshold:.3f}")
    axes[1].set(xlabel="False Positive Rate", ylabel="Recall", title="ROC-кривая",
                xlim=(0, 1), ylim=(0, 1.02))
    axes[1].legend()

    pr_precision, pr_recall, _ = metrics.precision_recall_curve(y_test, test_probabilities)
    axes[2].plot(pr_recall, pr_precision,
                 label=f"PR-AUC = {metrics.average_precision_score(y_test, test_probabilities):.3f}")
    axes[2].axhline(y_test.mean(), linestyle="--", color="grey", label="Случайный отбор")
    axes[2].scatter(recall, precision, color="crimson", s=65,
                    zorder=3, label=f"Порог {threshold:.3f}")
    axes[2].set(xlabel="Recall", ylabel="Precision", title="PR-кривая",
                xlim=(0, 1.02), ylim=(0, 1.02))
    axes[2].legend()
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    st.info(
        "Если увеличить порог, в подборку попадёт меньше вин: обычно растёт доля "
        "действительно хороших среди рекомендаций, но больше хороших вин будут пропущены. "
        "Если уменьшить порог, подборка станет шире, но посредственных вин в ней может стать больше."
    )

with batch_tab:
    st.subheader("Загрузка CSV")
    st.write("Загрузите CSV с теми же 12 признаками, что и у модели, без столбца target.")
    uploaded = st.file_uploader("Файл с винами", type="csv")
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
        except (OSError, ValueError, UnicodeError, pd.errors.ParserError) as exc:
            st.error(f"Не удалось прочитать CSV: {exc}")
        else:
            missing = [name for name in feature_names if name not in batch.columns]
            if missing:
                st.error("Не хватает столбцов: " + ", ".join(missing))
            elif batch.empty:
                st.error("CSV не содержит строк с винами.")
            else:
                numeric = batch.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
                if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
                    st.error("В признаках есть пустые, нечисловые или бесконечные значения.")
                else:
                    probabilities = model.predict_proba(numeric)[:, 1]
                    result = batch.copy()
                    result["probability_good"] = probabilities
                    result["recommend"] = probabilities >= threshold
                    st.dataframe(result, width="stretch")
                    st.download_button(
                        "Скачать предсказания CSV",
                        data=result.to_csv(index=False).encode("utf-8-sig"),
                        file_name="wine_predictions.csv",
                        mime="text/csv",
                    )

with weights_tab:
    st.subheader("Коэффициенты после стандартизации")
    coefficients = pd.Series(
        model.named_steps["logisticregression"].coef_[0], index=feature_names
    ).sort_values()
    colors = ["#b24b4b" if value < 0 else "#427aa1" for value in coefficients]
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(coefficients.index.tolist(), coefficients.to_numpy(), color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Коэффициент логистической регрессии", ylabel="Признак")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)
    st.write(
        f"Сильнее всего за хорошее вино голосует **{coefficients.idxmax()}** "
        f"(коэффициент {coefficients.max():+.2f}), а против — "
        f"**{coefficients.idxmin()}** ({coefficients.min():+.2f}). "
        "Знак коэффициента описывает влияние при прочих равных."
    )
