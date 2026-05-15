import io
import json
import zipfile
import numpy as np
import pandas as pd


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def save_session(dashboard_obj) -> bytes:
    state = {
        "version": "1.0",
        "config": {
            "location_column": dashboard_obj.location_column,
            "time_column": dashboard_obj.time_column,
            "value_column": dashboard_obj.value_column,
            "chart_column": dashboard_obj.chart_column,
            "filter_columns_names": dashboard_obj.filter_columns_names,
            "geo_data_name": dashboard_obj.geo_data_name,
            "geo_field_value": dashboard_obj.geo_feild_value,
            "skip_map_flag": dashboard_obj.skip_map_flag,
        },
        "filters": {
            "year_range": list(dashboard_obj.year_range.value) if dashboard_obj.time_column else None,
            "filter_values": {
                name: dashboard_obj.filter_columns_widgets[i].value
                for i, name in enumerate(dashboard_obj.filter_columns_names)
            },
        },
        "visualization": {
            "agg_type": dashboard_obj.agg_buttons.value,
            "value_column": dashboard_obj.select_value_column_update.value,
            "chart_x": dashboard_obj.select_chart_x_update.value,
            "legend": dashboard_obj.select_legend_update.value,
        },
        "map_settings": {
            "base_map": dashboard_obj.select_base_map.value,
            "color_map": dashboard_obj.select_color_map.value,
            "transparency": dashboard_obj.transparency_map_range.value,
            "tooltip_columns": dashboard_obj.select_tooltip.value,
        },
        "special_charts": {
            "heatmap_fields": dashboard_obj.select_heatmap_fields.value,
            "sankey_source": dashboard_obj.select_sankey_source.value,
            "sankey_target": dashboard_obj.select_sankey_target.value,
        },
        "chart_visibility": {
            "map": dashboard_obj.map_show.button_style == "solid",
            "bar": dashboard_obj.bar_show.button_style == "solid",
            "line": dashboard_obj.line_show.button_style == "solid",
            "box": dashboard_obj.box_show.button_style == "solid",
            "violin": dashboard_obj.violin_show.button_style == "solid",
            "scatter": dashboard_obj.scatter_show.button_style == "solid",
            "pie": dashboard_obj.pie_show.button_style == "solid",
            "radar": dashboard_obj.radar_show.button_style == "solid",
            "heatmap": dashboard_obj.heatmap_show.button_style == "solid",
            "sankey": dashboard_obj.sankey_show.button_style == "solid",
        },
        "analysis": {
            "clustering": dashboard_obj.clustering_show.value,
            "regression": dashboard_obj.regression_show.value,
            "classification": dashboard_obj.classification_show.value,
        },
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("state.json", json.dumps(state, indent=2, default=_json_default))

        csv_buf = io.StringIO()
        dashboard_obj.dataset.to_csv(csv_buf, index=False)
        zf.writestr("dataset.csv", csv_buf.getvalue())

        if dashboard_obj.geo_data_name == "uploaded":
            geo_features = dashboard_obj.geo_handler.get_single_geojson("uploaded")
            if geo_features:
                zf.writestr("geojson.json", json.dumps(geo_features))

    return buf.getvalue()


def load_session(zip_bytes: bytes) -> dict:
    buf = io.BytesIO(zip_bytes)
    with zipfile.ZipFile(buf, "r") as zf:
        state = json.loads(zf.read("state.json"))
        dataset = pd.read_csv(io.StringIO(zf.read("dataset.csv").decode("utf-8")))
        geojson = None
        if "geojson.json" in zf.namelist():
            geojson = json.loads(zf.read("geojson.json"))

    return {"state": state, "dataset": dataset, "geojson": geojson}
