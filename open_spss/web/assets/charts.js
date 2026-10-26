/* OpenSPSS 图表引擎:ECharts 惰性初始化与图表/单表导出绑定(由 app.js 拆分) */
"use strict";

let chartObservers = [];

function initChartIn(box, chart) {
  const opt = JSON.parse(JSON.stringify(chart.option));
  // 热力图 {c} 模板会输出整个 [x,y,v] 数组,替换为取值函数
  (opt.series || []).forEach(sr => {
    if (sr.type === "heatmap" && sr.label && sr.label.formatter === "{c}") {
      sr.label.formatter = (p) => {
        const v = Array.isArray(p.value) ? p.value[2] : p.value;
        return typeof v === "number" ? (+v.toFixed(4)).toString() : String(v ?? "");
      };
    }
  });
  const inst = echarts.getInstanceByDom(box) || echarts.init(box, null, { renderer: "canvas" });
  inst.clear();
  inst.setOption(opt);
  window.addEventListener("resize", () => inst.resize());
  return inst;
}

function chartPrefixToResult(prefix) {
  if (!prefix) return S.result;
  return S.history[parseInt(prefix.replace("rp", ""))]?.result || null;
}

function bindChartLazy(result = S.result, prefix = "") {
  $$(".chart-box").forEach(box => {
    const rest = box.id.startsWith("chart-rp") ? "rp" : "";
    const idx = parseInt(box.id.slice(rest ? 8 : 6));
    const chart = (rest ? chartPrefixToResult(box.id.slice(6)) : result)?.charts[idx];
    if (!chart) return;
    // 已在视口内的立即渲染;视口外的滚动到可见时再画(IntersectionObserver)
    const r = box.getBoundingClientRect();
    if (r.top < innerHeight + 80 && r.bottom > -80) {
      initChartIn(box, chart);
      return;
    }
    const io = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting) {
        io.disconnect();
        initChartIn(box, chart);
      }
    }, { rootMargin: "80px" });
    io.observe(box);
    chartObservers.push(io);
  });
  $$(".chart-tools [data-png]").forEach(b => b.addEventListener("click", () => {
    const box = b.closest(".rcard")?.querySelector(".chart-box");
    if (!box) return;
    const rest = box.id.startsWith("chart-rp");
    const ctx = rest ? chartPrefixToResult(box.id.slice(6).replace(/-\d+$/, "")) : S.result;
    const chart = ctx?.charts[parseInt(box.id.slice(rest ? 8 : 6))];
    if (!chart) return;
    const inst = initChartIn(box, chart);
    const url = inst.getDataURL({ pixelRatio: 2, backgroundColor: "#fff" });
    const a = document.createElement("a");
    a.href = url;
    const title = b.closest(".rcard").querySelector("h3")?.firstChild?.textContent?.trim() || "chart";
    a.download = title + ".png";
    a.click();
  }));
  $$(".dl-xlsx[data-tbl]").forEach(b => b.addEventListener("click", () => {
    const sec = b.closest(".report-item");
    const tables = sec ? (S.history[parseInt(sec.id.slice(2))]?.result.tables || []) : S.result.tables;
    const t = tables[parseInt(b.dataset.tbl)];
    if (t) exportXlsx([t]);
  }));
}
