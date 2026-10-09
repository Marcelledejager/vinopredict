"""VinoPredict: inference and evidence from a notebook-generated artifact bundle."""

from pathlib import Path
from importlib.metadata import version
import hashlib
import json
import platform

import altair as alt
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


st.set_page_config(page_title="VinoPredict", page_icon=":material/wine_bar:", layout="wide")

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
MAROON = "#58181F"
WINE = "#9C4B52"
CREAM = "#FAF6F3"

st.title(":material/wine_bar: VinoPredict")
st.caption("A quality forecast from your wine's laboratory measurements.")

# This app deliberately has no training code or dependency on the analysis folder.
# Read the manifest first so incomplete exports produce a useful setup message.
manifest_path = DATA_DIR / "manifest.json"
if not manifest_path.exists():
    with st.container(border=True):
        st.subheader("Add your notebook exports")
        st.write("Run the analysis notebook, then copy the contents of outputs/app_artifacts into this app's data folder.")
        st.write("Copy the exported requirements-app.txt to the app root as requirements.txt, install those dependencies, and restart the app.")
        st.caption("The app will use your fitted model and calculated results once the complete bundle is present.")
    st.stop()

try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    st.error(f"The export manifest cannot be read: {error}")
    st.stop()

required_files = [
    "model.joblib", "schema.json", "evaluation.json", "dataset_summary.json",
    "reference_profiles.csv", "feature_importance.csv", "model_comparison.csv",
    "calibration_curve.csv", "calibration_bins.csv", "roc_curve.csv",
    "precision_recall_curve.csv", "data_dictionary.csv", "example_inputs.csv", "requirements-app.txt",
]
missing_files = [name for name in required_files if not (DATA_DIR / name).is_file()]
if missing_files:
    st.error("The export bundle is incomplete. Copy the entire notebook export folder together.")
    st.code("\n".join(missing_files), language=None)
    st.stop()
if manifest.get("artifact_format_version") != 1:
    st.error("This app needs an artifact bundle in format version 1.")
    st.stop()

# Exact model-library versions avoid loading a pickle under another sklearn version.
version_mismatches = []
for package in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "shap"]:
    expected_version = manifest["packages"].get(package)
    installed_version = version(package)
    if expected_version != installed_version:
        version_mismatches.append(f"{package}: exported with {expected_version}; installed {installed_version}")
if version_mismatches:
    st.error("Install the requirements file exported by the notebook, then restart the app.")
    st.code("\n".join(version_mismatches), language=None)
    st.stop()
current_python_minor = ".".join(platform.python_version().split(".")[:2])
if current_python_minor != manifest["python_minor"]:
    st.warning(f"The notebook used Python {manifest['python_minor']}; this app uses {current_python_minor}. Use the same minor version for deployment.")

# Loading once per user session keeps the code linear, without custom cache functions.
# File signatures invalidate the session bundle after a new export is copied in.
bundle_signature = tuple(
    (name, (DATA_DIR / name).stat().st_mtime_ns, (DATA_DIR / name).stat().st_size)
    for name in ["manifest.json"] + required_files
)
if st.session_state.get("bundle_signature") != bundle_signature:
    with st.spinner("Loading your wine model…"):
        mismatched_files = []
        for name in required_files:
            expected_hash = manifest.get("files", {}).get(name, {}).get("sha256")
            actual_hash = hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                mismatched_files.append(name)
        if mismatched_files:
            st.error("Some files belong to a different or incomplete export. Copy the complete bundle again.")
            st.code("\n".join(mismatched_files), language=None)
            st.stop()
        try:
            schema = json.loads((DATA_DIR / "schema.json").read_text(encoding="utf-8"))
            evaluation = json.loads((DATA_DIR / "evaluation.json").read_text(encoding="utf-8"))
            dataset = json.loads((DATA_DIR / "dataset_summary.json").read_text(encoding="utf-8"))
            model = joblib.load(DATA_DIR / "model.joblib")
            references = pd.read_csv(DATA_DIR / "reference_profiles.csv")
            importance = pd.read_csv(DATA_DIR / "feature_importance.csv")
            model_comparison = pd.read_csv(DATA_DIR / "model_comparison.csv")
            roc_data = pd.read_csv(DATA_DIR / "roc_curve.csv")
            pr_data = pd.read_csv(DATA_DIR / "precision_recall_curve.csv")
            calibration_data = pd.read_csv(DATA_DIR / "calibration_curve.csv")
            calibration_bins = pd.read_csv(DATA_DIR / "calibration_bins.csv")
            dictionary = pd.read_csv(DATA_DIR / "data_dictionary.csv")
            example_inputs = pd.read_csv(DATA_DIR / "example_inputs.csv")
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")
        except Exception as error:
            st.error(f"The exported model bundle could not be loaded: {error}")
            st.stop()
        bundle_ids = {schema.get("bundle_id"), evaluation.get("bundle_id"), dataset.get("bundle_id"), manifest.get("bundle_id")}
        if len(bundle_ids) != 1:
            st.error("The model summaries are from different exports. Copy a complete bundle.")
            st.stop()
        if list(model.feature_names_in_) != schema["model_features"]:
            st.error("The input schema and model expect different measurements. Copy a complete bundle.")
            st.stop()
        st.session_state["bundle"] = {
            "schema": schema, "evaluation": evaluation, "dataset": dataset,
            "model": model, "explainer": explainer, "references": references,
            "importance": importance, "model_comparison": model_comparison,
            "roc": roc_data, "pr": pr_data, "calibration": calibration_data,
            "calibration_bins": calibration_bins, "dictionary": dictionary, "examples": example_inputs,
        }
        st.session_state["bundle_signature"] = bundle_signature
        st.session_state.pop("forecast", None)
        st.session_state.pop("batch_forecast", None)
        st.session_state.pop("batch_upload_hash", None)

bundle = st.session_state["bundle"]
schema = bundle["schema"]
evaluation = bundle["evaluation"]
dataset = bundle["dataset"]
model = bundle["model"]
references = bundle["references"].set_index("feature")
input_names = [item["name"] for item in schema["inputs"]]
model_features = schema["model_features"]
threshold = float(schema["target"]["threshold"])
positive_index = int(np.flatnonzero(model.classes_ == schema["target"]["positive_class"])[0])

with st.sidebar:
    st.subheader("Your wine workspace")
    page = st.radio("View", ["Forecast a wine", "Model evidence", "Dataset"], label_visibility="collapsed")
    st.caption(f"Model export: {manifest['created_utc'][:10]}")

if page == "Forecast a wine":
    # Numeric inputs follow the project brief and retain the laboratory units.
    with st.sidebar:
        st.subheader("Laboratory measurements")
        with st.form("wine_measurements"):
            batch_label = st.text_input("Wine or batch name", value="My wine", max_chars=100)
            measurement_values = {}
            for item in schema["inputs"]:
                measurement_values[item["name"]] = st.number_input(
                    f"{item['label']} ({item['unit']})",
                    min_value=float(item["minimum"]),
                    max_value=float(item["maximum"]) if item["maximum"] is not None else None,
                    value=float(item["default"]), step=float(item["step"]),
                    format=f"%.{item['decimals']}f", help=item["description"],
                    key=f"{manifest['bundle_id']}_{item['name']}",
                )
            submitted = st.form_submit_button("Forecast quality", type="primary", width="stretch")

    if submitted:
        input_frame = pd.DataFrame([measurement_values], columns=input_names)
        input_errors = []
        if not np.isfinite(input_frame.to_numpy()).all():
            input_errors.append("Enter finite values for every measurement.")
        if input_frame.loc[0, "density"] <= 0:
            input_errors.append("Density must be greater than zero.")
        if input_frame.loc[0, "free_sulfur_dioxide"] > input_frame.loc[0, "total_sulfur_dioxide"]:
            input_errors.append("Free sulfur dioxide cannot exceed total sulfur dioxide.")
        if input_errors:
            st.error(" ".join(input_errors))
            st.session_state.pop("forecast", None)
        else:
            model_input = input_frame.copy()
            if schema["representation"] == "ratios":
                epsilon = float(schema["ratio_epsilon"])
                model_input["acidity_ratio"] = model_input["fixed_acidity"] / model_input["volatile_acidity"].clip(lower=epsilon)
                model_input["free_total_so2_ratio"] = model_input["free_sulfur_dioxide"] / model_input["total_sulfur_dioxide"].clip(lower=epsilon)
            model_input = model_input[model_features]
            with st.spinner("Reading the chemistry…"):
                probability = float(model.predict_proba(model_input)[0, positive_index])
                shap_values = bundle["explainer"].shap_values(model_input)[0, :, positive_index]
                baseline = float(np.asarray(bundle["explainer"].expected_value)[positive_index])
            local_factors = pd.DataFrame({"feature": model_features, "value": model_input.iloc[0].to_numpy(), "contribution": shap_values})
            local_factors["label"] = local_factors["feature"].map(references["label"])
            local_factors["percentage_points"] = local_factors["contribution"] * 100
            local_factors["direction"] = np.where(local_factors["contribution"] >= 0, "Raises estimate", "Lowers estimate")
            outside_range = []
            for feature in model_features:
                value = float(model_input.loc[0, feature])
                if value < references.loc[feature, "training_min"] or value > references.loc[feature, "training_max"]:
                    outside_range.append(str(references.loc[feature, "label"]))
            st.session_state["forecast"] = {
                "name": batch_label.strip() or "My wine", "inputs": input_frame,
                "model_input": model_input, "probability": probability, "baseline": baseline,
                "factors": local_factors, "outside_range": outside_range,
            }

    forecast = st.session_state.get("forecast")
    if forecast is None:
        with st.container(border=True):
            st.subheader("Turn a laboratory profile into a quality estimate")
            st.write("Enter the eleven measurements in the sidebar and select Forecast quality.")
            st.caption("Defaults are development-data medians and do not represent a measured wine.")
        preview_columns = st.columns(3)
        preview_columns[0].metric("Chemical inputs", len(input_names))
        preview_columns[1].metric("Good wine definition", "Quality ≥ 6")
        preview_columns[2].metric("Held-out ROC-AUC", f"{evaluation['metrics']['roc_auc']:.3f}")
    else:
        st.subheader(forecast["name"])
        probability = forecast["probability"]
        predicted_class = "Good" if probability >= threshold else "Bad"
        result_columns = st.columns(3)
        with result_columns[0].container(border=True):
            st.metric("Estimated probability of Good", f"{probability:.1%}")
        with result_columns[1].container(border=True):
            st.metric("Predicted class", predicted_class)
            st.caption("Good means a sensory quality score of at least 6.")
        with result_columns[2].container(border=True):
            st.metric("Decision cutoff", f"{threshold:.0%}")
            st.caption("The cutoff was fixed before holdout evaluation.")
        st.caption("This is a model probability estimate. Its reliability is shown under Model evidence.")
        if forecast["outside_range"]:
            st.warning("Outside the observed development range: " + ", ".join(forecast["outside_range"]) + ". The prediction involves extrapolation.")

        factor_column, reference_column = st.columns([1, 1], gap="large")
        with factor_column.container(border=True):
            st.subheader("What influenced this estimate")
            factor_data = forecast["factors"].copy()
            factor_chart = alt.Chart(factor_data).mark_bar().encode(
                x=alt.X("percentage_points:Q", title="Contribution to Good probability (percentage points)"),
                y=alt.Y("label:N", sort="-x", title=None),
                color=alt.Color("direction:N", scale=alt.Scale(domain=["Raises estimate", "Lowers estimate"], range=[MAROON, WINE]), legend=alt.Legend(title=None, orient="bottom")),
                tooltip=[alt.Tooltip("label:N", title="Measurement"), alt.Tooltip("value:Q", format=".4f"), alt.Tooltip("percentage_points:Q", title="Contribution (pp)", format="+.2f")],
            )
            factor_zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color=MAROON).encode(x="zero:Q")
            st.altair_chart((factor_chart + factor_zero).properties(height=350), width="stretch")
            st.caption(f"Baseline {forecast['baseline']:.1%} + contributions = estimate {probability:.1%}. Contributions describe the model; they do not show the effect of changing a measurement.")
            negative_factors = factor_data.loc[factor_data["contribution"] < 0].sort_values("contribution").head(3)
            if len(negative_factors):
                st.write("**Largest factors lowering the estimate**")
                for row in negative_factors.itertuples(index=False):
                    reference = references.loc[row.feature]
                    st.write(f"**{row.label}**: {row.value:.4g} {reference['unit']}, contribution {row.percentage_points:+.1f} percentage points. Good-wine middle 50%: {reference['good_q25']:.4g}–{reference['good_q75']:.4g} {reference['unit']}.")

        with reference_column.container(border=True):
            st.subheader("Your wine and the observed references")
            comparison = references.loc[input_names, ["label", "unit", "training_min", "training_max", "good_q25", "good_median", "good_q75"]].copy()
            comparison["Your wine"] = forecast["inputs"].iloc[0]
            comparison = comparison.rename(columns={"label": "Measurement", "unit": "Unit", "good_q25": "Good Q25", "good_median": "Good median", "good_q75": "Good Q75"})
            st.dataframe(comparison[["Measurement", "Unit", "Your wine", "Good Q25", "Good median", "Good Q75"]].style.set_properties(**{"color": MAROON}), hide_index=True, width="stretch",
                         column_config={column: st.column_config.NumberColumn(format="%.4f") for column in ["Your wine", "Good Q25", "Good median", "Good Q75"]})
            st.caption("Good references are quartiles and medians from development wines scoring at least 6. They describe observed wines and do not establish optimum values.")
            selected_feature = st.selectbox("Inspect a measurement", input_names, format_func=references["label"].to_dict().get)
            selected_reference = references.loc[selected_feature]
            reference_band = pd.DataFrame({
                "measurement": [selected_reference["label"]], "low": [selected_reference["good_q25"]],
                "high": [selected_reference["good_q75"]], "median": [selected_reference["good_median"]],
                "input": [float(forecast["inputs"].loc[0, selected_feature])],
            })
            range_band = alt.Chart(reference_band).mark_rule(strokeWidth=14, color="#D9A5AA").encode(x=alt.X("low:Q", title=f"{selected_reference['label']} ({selected_reference['unit']})"), x2="high:Q", y=alt.Y("measurement:N", title=None, axis=None))
            median_marker = alt.Chart(reference_band).mark_tick(color=WINE, thickness=3, size=32).encode(x="median:Q", y="measurement:N")
            input_marker = alt.Chart(reference_band).mark_point(color=MAROON, filled=True, size=180).encode(x="input:Q", y="measurement:N", tooltip=[alt.Tooltip("input:Q", title="Your measurement", format=".4f")])
            st.altair_chart((range_band + median_marker + input_marker).properties(height=100), width="stretch")
            st.caption("Rose band: Good middle 50%. Wine mark: Good median. Maroon point: your wine.")

        prediction_export = forecast["inputs"].copy()
        prediction_export.insert(0, "wine_name", forecast["name"])
        prediction_export["probability_good"] = probability
        prediction_export["predicted_class"] = predicted_class
        prediction_export["decision_threshold"] = threshold
        prediction_export["bundle_id"] = manifest["bundle_id"]
        st.download_button("Download this forecast", prediction_export.to_csv(index=False).encode("utf-8"), file_name="vino_forecast.csv", mime="text/csv", icon=":material/download:")

    with st.expander("Forecast a CSV of wines", icon=":material/table_chart:"):
        st.write("Upload the same eleven measurement columns, in the same units. Quality is not required. Column names may use dots, spaces, or underscores.")
        st.download_button("Download example input CSV", bundle["examples"].to_csv(index=False).encode("utf-8"), file_name="vino_example_inputs.csv", mime="text/csv")
        uploaded_csv = st.file_uploader("Wine measurements CSV", type=["csv"], key="batch_upload")
        if uploaded_csv is not None:
            try:
                batch = pd.read_csv(uploaded_csv, sep=None, engine="python")
                batch.columns = batch.columns.str.strip().str.lower().str.replace(".", "_", regex=False).str.replace(" ", "_", regex=False)
            except Exception as error:
                st.error(f"The CSV could not be read: {error}")
                batch = None
            if batch is not None:
                missing_columns = [feature for feature in input_names if feature not in batch.columns]
                if batch.columns.duplicated().any():
                    st.error("The CSV has duplicate measurement names after normalization.")
                elif missing_columns:
                    st.error("Missing measurements: " + ", ".join(missing_columns))
                elif len(batch) == 0 or len(batch) > 5000:
                    st.error("Upload between 1 and 5,000 wines per file.")
                else:
                    batch_inputs = batch[input_names].apply(pd.to_numeric, errors="coerce")
                    valid_rows = pd.Series(np.isfinite(batch_inputs.to_numpy()).all(axis=1), index=batch.index)
                    valid_rows &= batch_inputs.ge(0).all(axis=1)
                    valid_rows &= batch_inputs["density"].gt(0)
                    valid_rows &= batch_inputs["ph"].le(14)
                    valid_rows &= batch_inputs["alcohol"].le(100)
                    valid_rows &= batch_inputs["free_sulfur_dioxide"].le(batch_inputs["total_sulfur_dioxide"])
                    if not valid_rows.all():
                        invalid_csv_lines = (np.flatnonzero(~valid_rows.to_numpy()) + 2).tolist()
                        st.error(f"Invalid measurements on CSV lines {invalid_csv_lines[:20]}. Values must be numeric and finite; concentrations cannot be negative; density must be positive; pH ≤ 14, alcohol ≤ 100%, and free SO₂ ≤ total SO₂.")
                    elif st.button("Forecast uploaded wines", type="primary"):
                        batch_model_input = batch_inputs.copy()
                        if schema["representation"] == "ratios":
                            epsilon = float(schema["ratio_epsilon"])
                            batch_model_input["acidity_ratio"] = batch_model_input["fixed_acidity"] / batch_model_input["volatile_acidity"].clip(lower=epsilon)
                            batch_model_input["free_total_so2_ratio"] = batch_model_input["free_sulfur_dioxide"] / batch_model_input["total_sulfur_dioxide"].clip(lower=epsilon)
                        batch_model_input = batch_model_input[model_features]
                        batch_probability = model.predict_proba(batch_model_input)[:, positive_index]
                        batch_output = batch.copy()
                        batch_output["probability_good"] = batch_probability
                        batch_output["predicted_class"] = np.where(batch_probability >= threshold, "Good", "Bad")
                        batch_output["decision_threshold"] = threshold
                        range_flags = batch_model_input.lt(references.loc[model_features, "training_min"]) | batch_model_input.gt(references.loc[model_features, "training_max"])
                        batch_output["outside_training_range"] = range_flags.any(axis=1)
                        batch_output["outside_range_measurements"] = [", ".join(batch_model_input.columns[row]) for row in range_flags.to_numpy()]
                        batch_output["bundle_id"] = manifest["bundle_id"]
                        st.session_state["batch_forecast"] = batch_output
                        st.session_state["batch_upload_hash"] = hashlib.sha256(uploaded_csv.getvalue()).hexdigest()
                upload_hash = hashlib.sha256(uploaded_csv.getvalue()).hexdigest()
                if st.session_state.get("batch_upload_hash") == upload_hash and "batch_forecast" in st.session_state:
                    batch_output = st.session_state["batch_forecast"]
                    st.dataframe(batch_output, hide_index=True, width="stretch", column_config={"probability_good": st.column_config.NumberColumn(format="percent")})
                    st.caption(f"{int(batch_output['outside_training_range'].sum())} wines include inputs outside the observed development range.")
                    st.download_button("Download batch forecasts", batch_output.to_csv(index=False).encode("utf-8"), file_name="vino_batch_forecasts.csv", mime="text/csv")

elif page == "Model evidence":
    st.subheader("How the exported model performed")
    st.caption(f"{evaluation['holdout_rows']:,} held-out rows in {evaluation['holdout_profiles']:,} unique chemical profiles. The deployed forest was fitted on {evaluation['development_rows']:,} development rows.")
    metric_columns = st.columns(4)
    for column, metric, label in zip(metric_columns, ["roc_auc", "balanced_accuracy", "f1_good", "brier_loss"], ["ROC-AUC", "Balanced accuracy", "Good-class F1", "Brier loss"]):
        with column.container(border=True):
            st.metric(label, f"{evaluation['metrics'][metric]:.3f}")
    st.caption("Higher is better for ROC-AUC, balanced accuracy and F1. Lower Brier loss is better.")
    curves_tab, decisions_tab, selection_tab, explanation_tab = st.tabs(["Ranking and reliability", "Class errors", "Model selection", "Model factors"])

    with curves_tab:
        curve_columns = st.columns(2)
        roc_chart = alt.Chart(bundle["roc"]).mark_line(color=MAROON, strokeWidth=3).encode(x=alt.X("false_positive_rate:Q", title="False positive rate", scale=alt.Scale(domain=[0, 1])), y=alt.Y("true_positive_rate:Q", title="True positive rate", scale=alt.Scale(domain=[0, 1])), order="false_positive_rate:Q")
        diagonal = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
        chance_line = alt.Chart(diagonal).mark_line(color=WINE, strokeDash=[6, 4]).encode(x="x:Q", y="y:Q")
        curve_columns[0].altair_chart((roc_chart + chance_line).properties(title="Held-out ROC curve", height=280), width="stretch")
        pr_chart = alt.Chart(bundle["pr"]).mark_line(color=MAROON, strokeWidth=3).encode(x=alt.X("recall:Q", title="Recall of Good", scale=alt.Scale(domain=[0, 1])), y=alt.Y("precision:Q", title="Precision of Good", scale=alt.Scale(domain=[0, 1])), order=alt.Order("recall:Q"))
        curve_columns[1].altair_chart(pr_chart.properties(title="Held-out precision–recall curve", height=280), width="stretch")
        calibration_chart = alt.Chart(bundle["calibration"]).mark_line(color=MAROON, point=True, strokeWidth=3).encode(x=alt.X("mean_predicted_probability:Q", title="Mean predicted probability", scale=alt.Scale(domain=[0, 1])), y=alt.Y("observed_good_proportion:Q", title="Observed Good proportion", scale=alt.Scale(domain=[0, 1])), tooltip=[alt.Tooltip("mean_predicted_probability:Q", format=".2f"), alt.Tooltip("observed_good_proportion:Q", format=".2f")])
        st.altair_chart((calibration_chart + chance_line).properties(title="Probability reliability", height=300), width="stretch")
        st.caption("The dashed line shows perfect reliability. The forest was not separately calibrated. This diagram uses the reserved holdout.")
        st.dataframe(bundle["calibration_bins"], hide_index=True, width="stretch")

    with decisions_tab:
        matrix = np.asarray(evaluation["confusion_matrix"])
        confusion_data = pd.DataFrame([
            {"Actual": actual, "Predicted": predicted, "Wines": int(matrix[i, j])}
            for i, actual in enumerate(["Bad", "Good"]) for j, predicted in enumerate(["Bad", "Good"])
        ])
        confusion_chart = alt.Chart(confusion_data).mark_rect().encode(x=alt.X("Predicted:N", sort=["Bad", "Good"]), y=alt.Y("Actual:N", sort=["Bad", "Good"]), color=alt.Color("Wines:Q", scale=alt.Scale(range=[CREAM, "#D9A5AA"]), legend=None))
        confusion_labels = alt.Chart(confusion_data).mark_text(color=MAROON, fontSize=26).encode(x=alt.X("Predicted:N", sort=["Bad", "Good"]), y=alt.Y("Actual:N", sort=["Bad", "Good"]), text="Wines:Q")
        st.altair_chart((confusion_chart + confusion_labels).properties(height=280), width="stretch")
        st.dataframe(pd.DataFrame(evaluation["metrics"].items(), columns=["Metric", "Value"]), hide_index=True, column_config={"Value": st.column_config.NumberColumn(format="%.3f")})
        intervals = pd.DataFrame(evaluation["confidence_intervals"])
        st.write("**95% intervals from resampling held-out profile groups**")
        st.dataframe(intervals, hide_index=True, width="stretch")
        st.caption("These intervals describe holdout sampling variation for the fitted model. They do not include model selection uncertainty or establish external performance.")

    with selection_tab:
        comparison_data = bundle["model_comparison"]
        comparison_chart = alt.Chart(comparison_data).mark_bar(color=MAROON).encode(x=alt.X("roc_auc_mean:Q", title="Mean grouped CV ROC-AUC", scale=alt.Scale(domain=[0, 1])), y=alt.Y("model:N", sort="-x", title=None), tooltip=["model:N", alt.Tooltip("roc_auc_mean:Q", format=".3f"), alt.Tooltip("roc_auc_std:Q", title="Fold SD", format=".3f")])
        st.altair_chart(comparison_chart.properties(height=300), width="stretch")
        st.dataframe(comparison_data, hide_index=True, width="stretch")
        st.write(f"**Exported representation:** {evaluation['representation']}. Original forest best CV ROC-AUC: {evaluation['original_cv_roc_auc']:.3f}; forest with ratios: {evaluation['ratios_cv_roc_auc']:.3f}.")
        st.caption("Ratios are retained only if they improve the tuned development CV score by more than 0.005. The brief's forest is the deployment candidate; linear benchmarks remain visible. The best tuning score is a selection estimate.")
        st.json(evaluation["parameters"])
        st.write("**Forest prediction formula**")
        st.latex(r"\widehat p(\mathrm{Good}\mid x) = \frac{1}{T}\sum_{t=1}^{T}p_t(\mathrm{Good}\mid x)")
        st.caption("Each tree contributes its weighted Good-class proportion in the leaf reached by the submitted measurements. The app labels Good at a probability of at least 0.50.")

    with explanation_tab:
        importance_data = bundle["importance"]
        importance_chart = alt.Chart(importance_data).mark_bar(color=MAROON).encode(x=alt.X("mean_absolute_shap:Q", title="Mean absolute probability contribution"), y=alt.Y("label:N", sort="-x", title=None), tooltip=["label:N", alt.Tooltip("mean_absolute_shap:Q", format=".4f"), alt.Tooltip("permutation_auc_drop:Q", title="Permutation AUC decrease", format=".4f")])
        st.altair_chart(importance_chart.properties(height=350), width="stretch")
        st.caption("Importance was calculated after selection on held-out predictions. Correlated measurements can share importance. SHAP explains associations learned by the forest; it does not identify chemical interventions.")

else:
    st.subheader("The wines behind the model")
    overview_columns = st.columns(4)
    overview_columns[0].metric("Source wines", f"{dataset['rows']:,}")
    overview_columns[1].metric("Chemical measurements", dataset["chemical_predictors"])
    overview_columns[2].metric("Observed quality scores", f"{dataset['quality_min']}–{dataset['quality_max']}")
    overview_columns[3].metric("Good in source", f"{dataset['good_percentage']:.1f}%")
    quality_distribution = pd.DataFrame({"Quality score": [int(score) for score in dataset["quality_counts"]], "Wines": list(dataset["quality_counts"].values())})
    count_chart = alt.Chart(quality_distribution).mark_bar(color=MAROON).encode(x=alt.X("Quality score:O", sort="ascending"), y="Wines:Q", tooltip=["Quality score:O", "Wines:Q"])
    count_labels = alt.Chart(quality_distribution).mark_text(color=MAROON, dy=-8).encode(x=alt.X("Quality score:O", sort="ascending"), y="Wines:Q", text="Wines:Q")
    st.altair_chart((count_chart + count_labels).properties(height=300), width="stretch")
    st.write(f"The source contains {dataset['duplicate_measurement_and_quality_rows']} duplicate measurement-and-quality rows and {dataset['unique_chemical_profiles']:,} unique chemical profiles. All repeats stay together in evaluation splits.")
    st.dataframe(bundle["dictionary"].style.set_properties(**{"color": MAROON}), hide_index=True, width="stretch")
    st.write("**Development references**")
    st.dataframe(bundle["references"][["label", "unit", "training_min", "training_max", "good_q25", "good_median", "good_q75"]], hide_index=True, width="stretch")
    st.caption("The data describes Portuguese red Vinho Verde wine. It does not contain producer, batch, vintage or location identifiers for broader generalization studies.")
    st.markdown("[Selected Kaggle source](https://www.kaggle.com/datasets/piyushgoyal443/red-wine-dataset) · [Original UCI context](https://archive.ics.uci.edu/dataset/186/wine+quality) · [Cortez et al. (2009)](https://doi.org/10.1016/j.dss.2009.05.016)")

st.caption("VinoPredict · chemistry, quality and a clear view of the evidence")
