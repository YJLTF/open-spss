/* OpenSPSS 图表引擎:ECharts 惰性初始化与图表/单表导出绑定(由 app.js 拆分) */
"use strict";

let chartObservers = [];

/* 所有图表实例统一响应窗口缩放(避免每个实例各挂一个 resize 监听) */
window.addEventListener("resize", () => {
  $$(".chart-box").forEach(b => echarts.getInstanceByDom(b)?.resize());
});

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
  return inst;
}

/* 图表 PNG 数据源:优先用已渲染实例直接导出;
   未渲染过的才初始化,并关闭动画——否则入场动画首帧会被导出成空数据图 */
function chartPngUrl(box, chart) {
  const existing = echarts.getInstanceByDom(box);
  if (existing) {
    return Promise.resolve(existing.getDataURL({ pixelRatio: 2, backgroundColor: "#fff" }));
  }
  return new Promise((resolve) => {
    const opt = JSON.parse(JSON.stringify(chart.option));
    opt.animation = false;
    const inst = echarts.init(box, null, { renderer: "canvas" });
    inst.setOption(opt);
    const url = inst.getDataURL({ pixelRatio: 2, backgroundColor: "#fff" });
    resolve(url);
  });
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
  $$(".chart-tools [data-png]").forEach(b => b.addEventListener("click", async () => {
    const box = b.closest(".rcard")?.querySelector(".chart-box");
    if (!box) return;
    const rest = box.id.startsWith("chart-rp");
    const ctx = rest ? chartPrefixToResult(box.id.slice(6).replace(/-\d+$/, "")) : S.result;
    const chart = ctx?.charts[parseInt(box.id.slice(rest ? 8 : 6))];
    if (!chart) return;
    const url = await chartPngUrl(box, chart);
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
