/* OpenSPSS 结果渲染:结果流 HTML 与 Excel 导出(由 app.js 拆分) */
"use strict";

function renderResult() {
  chartObservers.forEach(o => o.disconnect()); chartObservers = [];
  const r = S.result;
  const nTables = r.tables.length, nCharts = r.charts.length;
  $("#results").innerHTML = `
    <div class="result-head">
      <div class="result-tabs">
        <button data-tab="all" class="active">全部结果</button>
        <button data-tab="interp">智能解读</button>
        <button data-tab="tables">表格 ${nTables}</button>
        <button data-tab="charts">图表 ${nCharts}</button>
      </div>
      <span style="flex:1"></span>
      <button class="btn" id="btnExport">⬇ 导出 Excel</button>
    </div>
    <div class="result-flow" id="resultFlow">${resultFlowHtml(r, "all")}</div>`;

  $$(".result-tabs button").forEach(b => b.addEventListener("click", () => {
    $$(".result-tabs button").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    $("#resultFlow").innerHTML = resultFlowHtml(S.result, b.dataset.tab);
    bindChartLazy();
  }));
  $("#btnExport").addEventListener("click", exportXlsx);
  bindChartLazy();
}

function resultFlowHtml(r, tab, prefix = "") {
  let html = "";
  const cp = (n) => `chart-${prefix}${n}`;
  if ((tab === "all" || tab === "interp") && r.interpretation.length) {
    html += `<div class="card rcard interp-card" data-kind="interp">
      <h3><span class="hbar"></span>智能解读</h3>
      <div class="interp-body">${r.interpretation.map(p => `<p>${md(p)}</p>`).join("")}</div>
    </div>`;
  }
  if (r.notes?.length && (tab === "all" || tab === "interp")) {
    html += `<div class="card rcard" data-kind="interp"><h3><span class="hbar" style="background:var(--teal)"></span>附加说明</h3>
      <div class="interp-body">${r.notes.map(n => `<p>${md(n)}</p>`).join("")}</div></div>`;
  }
  if (tab === "all" || tab === "tables") {
    r.tables.forEach((t, i) => {
      html += `<div class="card rcard" data-kind="tables">
        <h3><span class="hbar"></span>${esc(t.title)}
          <span class="dl-xlsx btn ghost" data-tbl="${i}" title="导出此表">⬇ Excel</span></h3>
        <div class="tbl-scroll"><table class="rtable"><thead><tr>${
          t.headers.map(h => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${
          t.rows.map(row => `<tr>${row.map((cell, ci) => {
            const s = String(cell ?? "");
            const isP = /^[<>]?\d*\.?\d+[%*]*$/.test(s) && (s.includes("*") || ci > 0 && /^<?0?\.?\d*\**$/.test(s));
            const cls = /^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(s) ? "num" : "";
            return `<td class="${cls} ${s.match(/^\<?0\.\d+\*+$/) || s.includes("*") ? "pcell" : ""}">${pCell(s)}</td>`;
          }).join("")}</tr>`).join("")}</tbody></table></div>
        ${t.note ? `<div class="tbl-note">${esc(t.note)}</div>` : ""}
      </div>`;
    });
  }
  if (tab === "all" || tab === "charts") {
    r.charts.forEach((c, i) => {
      html += `<div class="card rcard" data-kind="charts">
        <h3><span class="hbar"></span>${esc(c.title)}
          <span class="chart-tools"><button data-png="${i}">⬇ PNG</button></span></h3>
        <div class="chart-box" id="${cp(i)}"></div>
      </div>`;
    });
  }
  if (!html) html = `<div class="result-empty card"><h4>该分类下暂无内容</h4></div>`;
  return html;
}

async function exportXlsx(tables = null) {
  const payload = {
    title: S.result?.model?.name || "分析结果",
    tables: (tables || S.result.tables).map(t => ({ title: t.title, headers: t.headers, rows: t.rows, note: t.note })),
  };
  try {
    const res = await fetch("/api/export/xlsx", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    if (!res.ok) throw new Error("导出失败");
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${payload.title}.xlsx`;
    a.click();
    URL.revokeObjectURL(a.href);
    toast("已导出 Excel", "ok");
  } catch (err) { toast(err.message, "err"); }
}
