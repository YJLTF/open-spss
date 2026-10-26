"""ECharts option 构建器:让各模型用几行代码生成统一风格的图表。"""
from __future__ import annotations

import math

import numpy as np

PALETTE = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#06b6d4",
           "#8b5cf6", "#ec4899", "#84cc16", "#f97316", "#14b8a6",
           "#3b82f6", "#a855f7", "#eab308", "#fb7185", "#2dd4bf"]

AXIS_STYLE = {
    "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
    "axisLabel": {"color": "#64748b", "fontSize": 11},
    "axisTick": {"lineStyle": {"color": "#cbd5e1"}},
    "splitLine": {"lineStyle": {"color": "#eef2f7"}},
}


def base(title: str, tooltip: str = "axis") -> dict:
    return {
        "title": {"text": title, "left": "center", "textStyle": {"fontSize": 13, "fontWeight": 600, "color": "#334155"}},
        "tooltip": {"trigger": tooltip},
        "legend": {"top": 26, "type": "scroll", "itemGap": 12,
                   "textStyle": {"color": "#64748b", "fontSize": 11}},
        "color": PALETTE,
        "grid": {"left": 56, "right": 28, "top": 64, "bottom": 48, "containLabel": True},
    }


def _cat_axis(name: str = "") -> dict:
    ax = {"type": "category", **AXIS_STYLE}
    ax["name"] = name
    return ax


def bar_chart(title: str, categories: list, series: dict[str, list],
              x_name: str = "", y_name: str = "", stack: bool = False,
              horizontal: bool = False) -> dict:
    opt = base(title)
    cat_axis = {**_cat_axis(x_name), "data": list(categories)}
    val_axis = {**AXIS_STYLE, "type": "value", "name": y_name}
    if horizontal:
        opt["xAxis"], opt["yAxis"] = val_axis, cat_axis
    else:
        opt["xAxis"], opt["yAxis"] = cat_axis, val_axis
    opt["series"] = [
        {"name": k, "type": "bar", "stack": "total" if stack else None,
         "data": list(v), "barMaxWidth": 42, "itemStyle": {"borderRadius": [4, 4, 0, 0]}}
        for k, v in series.items()
    ]
    return opt


def line_chart(title: str, x: list, series: dict[str, list],
               x_name: str = "", y_name: str = "", smooth: bool = True,
               area: bool = False, y_min=None, y_max=None) -> dict:
    opt = base(title)
    opt["xAxis"] = {**_cat_axis(x_name), "data": list(x), "boundaryGap": False}
    ya = {**AXIS_STYLE, "type": "value", "name": y_name}
    if y_min is not None:
        ya["min"] = y_min
    if y_max is not None:
        ya["max"] = y_max
    opt["yAxis"] = ya
    opt["series"] = [
        {"name": k, "type": "line", "data": v, "smooth": smooth, "symbolSize": 5,
         "lineStyle": {"width": 2.4},
         "areaStyle": {"opacity": 0.12} if area else None}
        for k, v in series.items()
    ]
    return opt


def scatter_chart(title: str, series: dict[str, list[list[float, float]]],
                  x_name: str = "", y_name: str = "", symbol: int = 7,
                  trend: list[list[float]] | None = None, trend_name: str = "拟合线") -> dict:
    opt = base(title, tooltip="item")
    opt["xAxis"] = {**AXIS_STYLE, "type": "value", "name": x_name, "scale": True}
    opt["yAxis"] = {**AXIS_STYLE, "type": "value", "name": y_name, "scale": True}
    opt["series"] = [
        {"name": k, "type": "scatter", "data": v, "symbolSize": symbol, "itemStyle": {"opacity": 0.72}}
        for k, v in series.items()
    ]
    if trend:
        opt["series"].append({
            "name": trend_name, "type": "line", "data": trend, "smooth": False,
            "showSymbol": False, "lineStyle": {"type": "dashed", "width": 2, "color": "#ef4444"},
            "tooltip": {"show": False},
        })
    return opt


def heatmap_chart(title: str, x_labels: list, y_labels: list,
                  values: list[list], x_name: str = "变量", fmt: str = "{c}",
                  v_min=None, v_max=None) -> dict:
    data = []
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            data.append([j, i, None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)])
    opt = base(title, tooltip="item")
    opt.pop("legend", None)
    opt["xAxis"] = {**_cat_axis(x_name), "data": x_labels, "axisLabel": {"rotate": 35, "fontSize": 10, "color": "#64748b"}, "splitArea": {"show": True}}
    opt["yAxis"] = {**_cat_axis(""), "data": y_labels, "splitArea": {"show": True}}
    opt["grid"] = {"left": 30, "right": 110, "top": 44, "bottom": 70, "containLabel": True}
    opt["visualMap"] = {
        "min": v_min if v_min is not None else float(np.nanmin(values)),
        "max": v_max if v_max is not None else float(np.nanmax(values)),
        "calculable": True, "orient": "vertical", "right": 4, "top": "center",
        "inRange": {"color": ["#3b82f6", "#93c5fd", "#f8fafc", "#fca5a5", "#ef4444"]},
        "textStyle": {"fontSize": 10, "color": "#64748b"},
    }
    opt["series"] = [{
        "type": "heatmap", "data": data, "label": {"show": True, "formatter": fmt, "fontSize": 9, "color": "#334155"},
        "emphasis": {"itemStyle": {"shadowBlur": 6, "shadowColor": "rgba(0,0,0,0.3)"}},
    }]
    return opt


def box_chart(title: str, groups: dict[str, list], y_name: str = "") -> dict:
    names, raw = list(groups.keys()), list(groups.values())
    stats = []
    for v in raw:
        arr = np.asarray([x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
        if arr.size == 0:
            stats.append([0, 0, 0, 0, 0])
            continue
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        iqr = q3 - q1
        lo = float(arr[arr >= q1 - 1.5 * iqr].min())
        hi = float(arr[arr <= q3 + 1.5 * iqr].max())
        stats.append([round(x, 4) for x in (lo, q1, med, q3, hi)])
    opt = base(title, tooltip="item")
    opt.pop("legend", None)
    opt["xAxis"] = {**_cat_axis(), "data": names}
    opt["yAxis"] = {**AXIS_STYLE, "type": "value", "name": y_name, "scale": True}
    opt["series"] = [
        {"name": "box", "type": "boxplot", "data": stats, "itemStyle": {"color": "#c7d2fe", "borderColor": "#6366f1"}},
        {"name": "散点", "type": "scatter",
         "data": [[i, float(np.random.choice(v))] for i, v in enumerate(raw) if len(v)],
         "itemStyle": {"color": "#94a3b8", "opacity": 0.35}, "symbolSize": 5, "tooltip": {"show": False}},
    ]
    return opt


def pie_chart(title: str, pairs: list[tuple], donut: bool = True) -> dict:
    opt = base(title, tooltip="item")
    opt.pop("legend", None)
    opt["tooltip"] = {"trigger": "item", "formatter": "{b}: {c} ({d}%)"}
    opt["series"] = [{
        "type": "pie", "radius": ["42%", "68%"] if donut else "68%",
        "center": ["50%", "56%"], "data": [{"name": k, "value": v} for k, v in pairs],
        "label": {"formatter": "{b}\n{d}%", "fontSize": 11, "color": "#475569"},
        "itemStyle": {"borderRadius": 6, "borderColor": "#fff", "borderWidth": 2},
    }]
    return opt


def hist_chart(title: str, values, x_name: str = "", normal_fit: bool = True,
               bins: int | None = None) -> dict:
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if bins is None:
        bins = max(6, min(24, int(np.ceil(np.sqrt(max(len(arr), 2))))))
    counts, edges = np.histogram(arr, bins=bins)
    labels = [f"{edges[i]:.2f}~{edges[i+1]:.2f}" for i in range(len(counts))]
    opt = base(title)
    opt.pop("legend", None)
    opt["xAxis"] = {**_cat_axis(x_name), "data": labels, "axisLabel": {"rotate": 40, "fontSize": 9, "color": "#64748b"}}
    opt["yAxis"] = {**AXIS_STYLE, "type": "value", "name": "频数"}
    opt["series"] = [{
        "type": "bar", "data": [int(c) for c in counts], "barCategoryGap": "8%",
        "itemStyle": {"color": "#818cf8", "borderRadius": [3, 3, 0, 0]},
        "tooltip": {"formatter": "{b}<br/>频数: {c}"},
    }]
    if normal_fit and arr.size > 2:
        mu, sd = arr.mean(), arr.std(ddof=1)
        width = edges[1] - edges[0]
        xs = np.linspace(edges[0], edges[-1], 200)
        pdf = (1 / (sd * math.sqrt(2 * math.pi))) * np.exp(-((xs - mu) ** 2) / (2 * sd ** 2)) * len(arr) * width
        opt["series"].append({
            "type": "line", "data": list(zip(labels_x(xs, edges), np.round(pdf, 2))),
            "showSymbol": False, "smooth": True, "lineStyle": {"color": "#ef4444", "width": 2},
            "tooltip": {"show": False},
        })
    return opt


def labels_x(xs, edges):
    idx = np.clip(np.searchsorted(edges, xs, side="right") - 1, 0, len(edges) - 2)
    return [f"{edges[i]:.2f}~{edges[i+1]:.2f}" for i in idx]


def qq_chart(title: str, values, dist="norm") -> dict:
    """Q-Q 图:理论分位数 vs 样本分位数,含 45 度参考线。"""
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    from scipy import stats as sps
    n = len(arr)
    p = (np.arange(1, n + 1) - 0.5) / n
    if dist == "norm":
        theo = sps.norm.ppf(p, loc=arr.mean(), scale=arr.std(ddof=1))
    else:
        theo = sps.uniform.ppf(p, loc=arr.min(), scale=arr.max() - arr.min())
    sample = np.sort(arr)
    pts = [[round(float(a), 4), round(float(b), 4)] for a, b in zip(theo, sample)]
    lo = float(min(theo.min(), sample.min()))
    hi = float(max(theo.max(), sample.max()))
    ref = [[lo, lo], [hi, hi]]
    opt = scatter_chart(title, {"样本分位数": pts}, x_name="理论分位数", y_name="样本分位数",
                        trend=ref, trend_name="参考线")
    return opt


def waterfall_chart(title: str, items: list[tuple[str, float]], y_name: str = "贡献") -> dict:
    """瀑布图(用于 TOPSIS 得分、权重展示等)"""
    cats = [k for k, _ in items]
    vals = [round(v, 4) for _, v in items]
    opt = base(title)
    opt.pop("legend", None)
    opt["xAxis"] = {**_cat_axis(), "data": cats, "axisLabel": {"rotate": 30, "fontSize": 10, "color": "#64748b"}}
    opt["yAxis"] = {**AXIS_STYLE, "type": "value", "name": y_name}
    opt["series"] = [
        {"type": "bar", "data": vals, "barMaxWidth": 46,
         "itemStyle": {"borderRadius": [4, 4, 0, 0],
                       "color": {"type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                                 "colorStops": [{"offset": 0, "color": "#818cf8"}, {"offset": 1, "color": "#6366f1"}]}},
         "label": {"show": True, "position": "top", "fontSize": 10, "color": "#475569", "formatter": "{c}"}},
    ]
    return opt


def radar_chart(title: str, indicators: list[tuple[str, float]],
                series: dict[str, list]) -> dict:
    """雷达图:indicators=[(维度名, 上限值)],series={系列名: [各维度数值]}"""
    opt = base(title)
    opt["radar"] = {
        "indicator": [{"name": n, "max": mx} for n, mx in indicators],
        "radius": "62%", "center": ["50%", "56%"],
        "axisName": {"color": "#475569", "fontSize": 11},
        "splitArea": {"areaStyle": {"color": ["#f8fafc", "#f1f5f9"]}},
        "splitLine": {"lineStyle": {"color": "#e2e8f0"}},
    }
    opt["series"] = [{
        "type": "radar",
        "data": [{"name": k, "value": v, "areaStyle": {"opacity": 0.15}} for k, v in series.items()],
        "symbolSize": 5, "lineStyle": {"width": 2.2},
    }]
    return opt


def pareto_chart(title: str, pairs: list[tuple[str, float]], y_name: str = "频数") -> dict:
    """帕累托图:降序柱 + 累计百分比折线(双 y 轴)"""
    total = sum(v for _, v in pairs) or 1
    cum, acc = [], 0.0
    for _, v in pairs:
        acc += v
        cum.append(round(acc / total * 100, 2))
    opt = base(title)
    opt["xAxis"] = {**_cat_axis(), "data": [k for k, _ in pairs], "axisLabel": {"rotate": 25, "fontSize": 10, "color": "#64748b"}}
    opt["yAxis"] = [
        {**AXIS_STYLE, "type": "value", "name": y_name},
        {**AXIS_STYLE, "type": "value", "name": "累计%", "max": 100, "min": 0,
         "axisLabel": {"formatter": "{value}%"}},
    ]
    opt["series"] = [
        {"name": y_name, "type": "bar", "data": [v for _, v in pairs],
         "barMaxWidth": 40, "itemStyle": {"borderRadius": [4, 4, 0, 0]}},
        {"name": "累计百分比", "type": "line", "yAxisIndex": 1, "data": cum,
         "smooth": False, "symbolSize": 5, "lineStyle": {"width": 2.2, "color": "#f59e0b"},
         "itemStyle": {"color": "#f59e0b"}},
    ]
    return opt


def funnel_chart(title: str, pairs: list[tuple[str, float]]) -> dict:
    opt = base(title, tooltip="item")
    opt.pop("legend", None)
    opt["series"] = [{
        "type": "funnel", "left": "12%", "right": "12%", "top": 52, "bottom": 16,
        "sort": "descending", "gap": 3, "minSize": "12%",
        "label": {"show": True, "position": "inside", "formatter": "{b}: {c}", "fontSize": 11},
        "itemStyle": {"borderColor": "#fff", "borderWidth": 2, "borderRadius": 3},
        "data": [{"name": k, "value": v} for k, v in pairs],
    }]
    return opt
