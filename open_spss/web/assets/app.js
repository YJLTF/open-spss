/* OpenSPSS 前端应用:视图/路由/交互(基础与图表能力见 core.js / charts.js / render.js) */
"use strict";

/* ---------------- 数据加载 ---------------- */
async function loadCatalog() { S.catalog = await api("/api/catalog"); renderTree($("#modelSearch").value); }
async function loadDatasets() {
  const d = await api("/api/datasets");
  S.datasets = d.datasets; S.samples = d.samples;
  renderDatasetBox();
}

async function selectDataset(id, { silent = false } = {}) {
  S.ds = await api(`/api/datasets/${id}?page=1`);
  localStorage.setItem("ospss.ds", id);
  S.dataPage = 1;
  renderDatasetBox();
  if (!silent) toast(`已切换数据集：${S.ds.name}`, "ok");
  const m = location.hash.match(/^#\/m\/([\w-]+)/);
  if (m) route();
  else location.hash = `#/data/${id}`;
}

async function ensureDs() {
  if (S.ds) return S.ds;
  const saved = localStorage.getItem("ospss.ds");
  if (saved && S.datasets.some(d => d.id === saved)) { S.ds = await api(`/api/datasets/${saved}?page=1`); return S.ds; }
  if (S.datasets.length) { S.ds = await api(`/api/datasets/${S.datasets[0].id}?page=1`); return S.ds; }
  return null;
}

/* ---------------- 侧栏 ---------------- */
const CAT_ICONS = {
  "通用统计": "M4 19h16M7 16v-5m5 5V8m5 8v-3",
  "差异检验": "M5 12h14M12 5v14",
  "回归建模": "M4 18l5-6 4 3 7-9",
  "降维聚类": "M12 12m-3 0a3 3 0 1 0 6 0 3 3 0 1 0-6 0M5 5m-2 0a2 2 0 1 0 4 0 2 2 0 1 0-4 0M19 6m-2 0a2 2 0 1 0 4 0 2 2 0 1 0-4 0M18 18m-2 0a2 2 0 1 0 4 0 2 2 0 1 0-4 0",
  "问卷研究": "M9 12l2 2 4-4m-3-6a9 9 0 1 0 .001 18.001A9 9 0 0 0 12 3z",
  "综合评价": "M12 3l2.6 5.6 6.4.8-4.7 4.3 1.2 6.3-5.5-3-5.5 3 1.2-6.3L3 9.4l6.4-.8z",
  "机器学习": "M12 3a4 4 0 0 1 4 4c2.5.5 4 2.5 4 5a5 5 0 0 1-5 5H9a5 5 0 0 1-5-5c0-2.5 1.5-4.5 4-5a4 4 0 0 1 4-4zM9 21h6",
  "时间预测": "M3 17l5-5 4 3 8-9m0 0h-5m5 0v5",
  "医学统计": "M12 21C7 17 3 13.5 3 9.5A4.5 4.5 0 0 1 12 7a4.5 4.5 0 0 1 9 2.5C21 13.5 17 17 12 21z",
  "质量管理": "M4 4v16h16M8 16v-5m4 5V8m4 8v-3",
  "统计绘图": "M5 20V10m7 10V4m7 16v-7",
};

function renderTree(filter = "") {
  const nav = $("#modelTree");
  if (!S.catalog) { nav.innerHTML = ""; return; }
  const kw = filter.trim().toLowerCase();
  let html = "";
  let shown = 0;
  for (const cat of S.catalog.categories) {
    const models = cat.models.filter(m =>
      !kw || m.name.toLowerCase().includes(kw) || (m.tags || []).some(t => t.toLowerCase().includes(kw)));
    if (!models.length) continue;
    shown += models.length;
    html += `<div class="cat-head"><span class="cat-dot"></span>${esc(cat.name)}</div>`;
    for (const m of models) {
      const active = S.model && S.model.id === m.id;
      html += `<button class="model-item ${active ? "active" : ""}" data-id="${m.id}" title="${esc(m.desc)}">
        <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="${CAT_ICONS[cat.name] || CAT_ICONS["通用统计"]}"/></svg>
        ${esc(m.name)}${m.demoNote ? `<span class="mi-tag">示例</span>` : ""}</button>`;
    }
  }
  if (!shown) html = `<div style="padding:20px 12px;font-size:12.5px;color:rgba(151,162,190,.7)">未找到匹配的模型</div>`;
  nav.innerHTML = html;
  $$(".model-item", nav).forEach(b => b.addEventListener("click", () => { location.hash = `#/m/${b.dataset.id}`; }));
}

function renderDatasetBox() {
  const box = $("#datasetBox");
  if (!S.datasets.length && !S.ds) {
    box.innerHTML = `<button class="ds-switch" id="dsGo"><span class="ds-name">⊕ 载入数据</span><div class="ds-meta">示例数据一键体验</div></button>`;
    $("#dsGo").addEventListener("click", () => { location.hash = "#/"; });
    return;
  }
  const cur = S.ds;
  box.innerHTML = `<button class="ds-switch" id="dsSwitch" title="切换数据集">
      <span class="ds-name"><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5.5" rx="8" ry="3"/><path d="M4 5.5V12c0 1.7 3.6 3 8 3s8-1.3 8-3V5.5M4 12v6.5c0 1.7 3.6 3 8 3s8-1.3 8-3V12"/></svg>
        <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(cur ? cur.name : "选择数据集")}</span></span>
      <div class="ds-meta">${cur ? `${cur.rows} 行 × ${cur.cols} 列` : "点击选择"}</div>
    </button>`;
  $("#dsSwitch").addEventListener("click", (e) => {
    const items = S.datasets.map(d => `
      <button class="ds-menu-item" data-id="${d.id}">📊 <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(d.name)}</span>
        <span class="ds-size">${d.rows}×${d.cols}</span>
        <span class="del" data-del="${d.id}" title="删除">×</span></button>`).join("")
      || `<div style="padding:10px;font-size:12px">暂无数据集</div>`;
    const pop = openPopover(e.currentTarget, `<div class="pop-head">数据集<button class="btn ghost" style="padding:2px 8px;font-size:11.5px" id="dsNew">＋ 上传/示例</button></div><div class="pop-list">${items}</div>`);
    $$(".ds-menu-item", pop).forEach(b => b.addEventListener("click", (ev) => {
      if (ev.target.dataset.del) { delDataset(ev.target.dataset.del); ev.stopPropagation(); return; }
      selectDataset(b.dataset.id); closePopover();
    }));
    $("#dsNew", pop)?.addEventListener("click", () => { closePopover(); location.hash = "#/"; });
  });
}

async function delDataset(id) {
  const d = S.datasets.find(x => x.id === id);
  if (!d) return;
  if (!confirm(`删除数据集「${d.name}」？将同时删除本地落盘文件，不可恢复。`)) return;
  try {
    await api(`/api/datasets/${id}`, { method: "DELETE" });
  } catch (err) { toast(err.message, "err"); return; }
  toast(`已删除数据集「${d.name}」`);
  if (S.ds && S.ds.id === id) { S.ds = null; localStorage.removeItem("ospss.ds"); }
  await loadDatasets();
  renderTree($("#modelSearch").value);
  route();
}

/* ---------------- 路由与顶栏 ---------------- */
function route() {
  closePopover(); closeModal();
  renderSideNav();
  const hash = location.hash || "#/";
  let m;
  if ((m = hash.match(/^#\/m\/([\w-]+)/))) return viewModel(m[1]);
  if ((m = hash.match(/^#\/data\/([\w-]+)/))) return viewData(m[1]);
  if (hash === "#/report") return viewReport();
  if ((m = hash.match(/^#\/plan\/([\w-]+)/))) return viewPlan(m[1]);
  viewWelcome();
}

function setTopbar(cat, name, right = "") {
  $("#topbar").innerHTML = `<div class="crumb">${cat ? `<span class="crumb-cat">${esc(cat)}</span>` : ""}<h1>${esc(name)}</h1></div>
    <div class="topbar-right">${right}</div>`;
}

/* ---------------- 欢迎页 ---------------- */
function viewWelcome() {
  S.model = null;
  setTopbar("", "开始分析");
  renderTree($("#modelSearch").value);
  const v = $("#view");
  const samples = S.samples.map((s, i) => `
    <div class="sample-card" data-key="${s.key}" tabindex="0" role="button">
      <svg class="sc-go" viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14m-6-6 6 6-6 6"/></svg>
      <h5>${esc(s.label.split("(")[0])}</h5><p>${esc(s.label)}<br>点击载入并开始体验</p>
    </div>`).join("");
  v.innerHTML = `
  <div class="welcome">
    <div class="hero">
      <div>
        <h2>把数据放进来，<br><span class="grad">剩下的交给统计。</span></h2>
        <p>OpenSPSS 内置 <b>${S.catalog?.total ?? "…"} 个分析模型</b>：从描述统计、假设检验到回归建模、机器学习与综合评价。选择模型、拖入变量、一键得到结果与智能解读。</p>
      </div>
      <div class="hero-badge">
        <div class="hb-item"><div class="hb-num">${S.catalog ? S.catalog.total : "—"}</div><div class="hb-lab">分析模型</div></div>
        <div class="hb-item"><div class="hb-num">${S.catalog ? S.catalog.categories.length : "—"}</div><div class="hb-lab">方法大类</div></div>
        <div class="hb-item"><div class="hb-num">100%</div><div class="hb-lab">本地运行</div></div>
      </div>
    </div>

    <div class="sec-title">上传数据</div>
    <div class="upload-card" id="dropzone" tabindex="0" role="button" aria-label="上传数据文件">
      <div class="uc-icon"><svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 16V4m0 0L7 9m5-5 5 5M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/></svg></div>
      <h4>拖拽文件到这里，或点击选择</h4>
      <p>支持 csv / xlsx / xls / sav（SPSS）/ tsv / json / parquet，60MB 以内</p>
      <input type="file" id="fileInput" hidden accept=".csv,.xlsx,.xls,.sav,.tsv,.json,.parquet">
    </div>

    <div class="sec-title">示例数据，即点即用</div>
    <div class="sample-grid">${samples}</div>

    <div class="sec-title">我的数据集</div>
    <div class="ds-list">${
      S.datasets.map(d => `
        <div class="ds-row" data-id="${d.id}" tabindex="0" role="button">
          <div><div class="dsr-name">${esc(d.name)} ${d.source === "sample" ? `<span class="badge-sample">示例</span>` : ""}</div>
          <div class="dsr-meta">${d.rows} 行 × ${d.cols} 列</div></div>
          <span class="del" data-delrow="${d.id}" title="删除">×</span>
          <svg class="arrow" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 6l6 6-6 6"/></svg>
        </div>`).join("") || `<p style="color:var(--ink-3);font-size:13px">暂无，载入示例或上传文件后出现在这里。</p>`
    }</div>
  </div>`;

  const dz = $("#dropzone"), fi = $("#fileInput");
  dz.addEventListener("click", () => fi.click());
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter") fi.click(); });
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("drag"); if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]); });
  fi.addEventListener("change", () => { if (fi.files[0]) uploadFile(fi.files[0]); });
  $$(".sample-card", v).forEach(c => c.addEventListener("click", async () => {
    const exist = S.datasets.find(d => d.source === "sample" && d.sampleKey === c.dataset.key);
    if (exist) {
      await selectDataset(exist.id);
      toast(`示例「${exist.name}」已在列表中，已直接切换；数据页可「↺ 恢复示例」重置`, "ok");
      return;
    }
    c.style.opacity = ".6";
    try {
      const ds = await api(`/api/datasets/sample/${c.dataset.key}`, { method: "POST" });
      await loadDatasets();
      await selectDataset(ds.id);
      if (ds.evicted) toast(`数据集已达上限（30 个），最早的「${ds.evicted}」已被清理`, "err");
    } catch (err) { toast(err.message, "err"); c.style.opacity = "1"; }
  }));
  $$(".ds-row", v).forEach(r => r.addEventListener("click", (e) => {
    const del = e.target.closest(".del[data-delrow]");
    if (del) { e.stopPropagation(); delDataset(del.dataset.delrow); return; }
    selectDataset(r.dataset.id);
  }));
}

async function uploadFile(file) {
  toast(`正在上传 ${file.name}…`);
  const fd = new FormData();
  fd.append("file", file);
  try {
    const ds = await api("/api/upload", { method: "POST", body: fd });
    await loadDatasets();
    await selectDataset(ds.id, { silent: true });
    if (ds.evicted) toast(`数据集已达上限（30 个），最早的「${ds.evicted}」已被清理`, "err");
    toast(`已载入「${ds.name}」：${ds.rows} 行 × ${ds.cols} 列`, "ok");
  } catch (err) { toast(err.message, "err"); }
}

/* ---------------- 数据页 ---------------- */
async function viewData(id) {
  S.model = null;
  try { S.ds = await api(`/api/datasets/${id}?page=${S.dataPage}&page_size=${S.pageSize}`); }
  catch { location.hash = "#/"; return; }
  renderTree($("#modelSearch").value);
  renderDataView();
}

function renderDataView() {
  const ds = S.ds;
  setTopbar("数据", ds.name, `
    <span class="chip-note">${ds.rows} 行 × ${ds.cols} 列</span>
    <button class="btn" id="btnUndo" ${ds.canUndo ? "" : "disabled"} title="撤销 (Ctrl+Z)">⤺ 撤销</button>
    <button class="btn" id="btnRedo" ${ds.canRedo ? "" : "disabled"} title="重做 (Ctrl+Y)">⤻ 重做</button>
    <button class="btn" id="btnMerge" title="与另一数据集纵向追加 / 横向关联">⧉ 合并</button>
    <button class="btn" id="btnExportDs" title="下载当前数据集（含编辑与模型写回的派生列）">⬇ 导出数据</button>
    <button class="btn" id="btnDsName" title="重命名数据集">✎ 重命名</button>
    ${ds.sampleKey ? `<button class="btn" id="btnResetSample" title="恢复为初始示例数据，当前修改将丢弃">↺ 恢复示例</button>` : ""}
    <button class="btn ${ds.weightCol ? "primary" : ""}" id="btnWeight" title="加权个案：按某列取值加权，描述/频数/卡方优先支持">⚖ 加权${ds.weightCol ? "：" + esc(ds.weightCol) : ""}</button>
    <button class="btn" id="btnProc">⚙ 数据处理</button>
    <button class="btn primary" id="btnAnalyze">开始分析 →</button>`);
  $("#btnAnalyze").addEventListener("click", () => { location.hash = "#/"; });
  $("#btnProc").addEventListener("click", openProcessModal);
  $("#btnUndo").addEventListener("click", doUndo);
  $("#btnRedo").addEventListener("click", doRedo);
  $("#btnMerge").addEventListener("click", openMergeModal);
  $("#btnExportDs")?.addEventListener("click", (e) => exportDsMenu(e.currentTarget));
  $("#btnDsName")?.addEventListener("click", openRenameDsModal);
  $("#btnResetSample")?.addEventListener("click", async () => {
    if (!confirm(`把「${ds.name}」恢复为初始示例数据？当前编辑与派生列将丢弃。`)) return;
    try {
      await api(`/api/datasets/${ds.id}/reset-sample`, { method: "POST" });
      S.ds = await api(`/api/datasets/${ds.id}?page=${S.dataPage}&page_size=${S.pageSize}`);
      renderDataView();
      toast("已恢复初始示例数据", "ok");
    } catch (err) { toast(err.message, "err"); }
  });
  $("#btnWeight").addEventListener("click", () => weightMenu($("#btnWeight")));

  const p = ds.preview;
  const colMeta = Object.fromEntries(ds.columns.map(c => [c.name, c]));
  const head = p.columns.map(c => {
    const meta = colMeta[c];
    const lab = meta.label ? ` title="变量标签：${esc(meta.label)}"` : "";
    return `<th${lab} data-col="${esc(c)}" title="点击修改变量类型">${esc(c)}<br>${vtBadge(meta.dtype)}</th>`;
  }).join("");
  const body = p.rows.map((row, i) => {
    const ridx = (p.page - 1) * p.pageSize + i;
    return `<tr><td class="rowidx">${ridx + 1}<span class="del-row" data-row="${ridx}" title="删除此行">×</span></td>${row.map((cell, ci) => {
      const isNum = typeof cell === "number";
      const meta = colMeta[p.columns[ci]];
      const vl = meta && meta.vals && meta.vals[String(cell)] ? ` title="值标签：${esc(meta.vals[String(cell)])}"` : "";
      return `<td${vl} data-row="${ridx}" data-col="${esc(p.columns[ci])}" class="${isNum ? "num" : ""}${cell === null ? " empty" : ""}">${cell === null ? "—" : esc(cell)}</td>`;
    }).join("")}</tr>`;
  }).join("");

  const chips = ds.columns.map(c => `
    <span class="col-chip" data-col="${esc(c.name)}" title="${esc(c.label || "点击打开变量设置")}">
      <b>${esc(c.name)}</b>${c.label ? `<i class="chip-label">${esc(c.label)}</i>` : ""}${vtBadge(c.dtype)}
    </span>`).join("");

  $("#view").innerHTML = `
  <div class="page-pad">
    <div class="toolbar">
      <span class="tb-title">数据工作台</span>
      <span class="tb-meta">双击单元格编辑 · ${ds.source === "sample" ? "示例数据" : "上传数据"}</span>
      <span class="spacer"></span>
      <button class="btn ghost" id="btnAddRow">＋ 行</button>
    </div>
    <div class="card data-table-wrap">
      <table class="grid"><thead><tr><th class="rowidx">#</th>${head}</tr></thead><tbody>${body}</tbody></table>
      <div class="pager">
        <span>共 ${p.total} 行 · 第 ${p.page} / ${Math.max(1, Math.ceil(p.total / p.pageSize))} 页</span>
        <select id="pgSize" title="每页行数">
          ${[50, 100, 200, 500].map(n => `<option value="${n}" ${S.pageSize === n ? "selected" : ""}>${n} 行/页</option>`).join("")}
        </select>
        <button id="pgPrev" ${p.page <= 1 ? "disabled" : ""}>上一页</button>
        <button id="pgNext" ${p.page * p.pageSize >= p.total ? "disabled" : ""}>下一页</button>
      </div>
    </div>
    <div class="card col-panel">
      <h4>变量设置 <span style="font-weight:400;font-size:12px;color:var(--ink-3)">点击变量：切换类型 · 编辑变量/值标签 · 删除列</span></h4>
      <div class="col-chips">${chips}</div>
    </div>
  </div>`;

  $("#pgSize").addEventListener("change", (e) => {
    S.pageSize = +e.target.value;
    localStorage.setItem("ospss.psize", e.target.value);
    S.dataPage = 1;
    viewData(ds.id);
  });
  $("#pgPrev").addEventListener("click", () => { S.dataPage--; viewData(ds.id); });
  $("#pgNext").addEventListener("click", () => { S.dataPage++; viewData(ds.id); });
  $("#btnAddRow").addEventListener("click", () => editSubmit({ action: "add_row" }));
  $$(".del-row").forEach(x => x.addEventListener("click", (e) => {
    e.stopPropagation();
    if (confirm(`删除第 ${+x.dataset.row + 1} 行？`)) editSubmit({ action: "del_row", row: +x.dataset.row });
  }));
  $$("th[data-col]").forEach(th => th.addEventListener("click", () => cycleType(th.dataset.col)));
  $$(".col-chip").forEach(ch => ch.addEventListener("click", () => variableMenu(ch)));
  $$(".grid tbody td[data-col]").forEach(td => td.addEventListener("dblclick", () => startCellEdit(td)));
}

/* ---------------- 数据工作台:编辑 / 撤销 / 标签 / 合并 ---------------- */
async function editSubmit(body) {
  try {
    S.ds = await api(`/api/datasets/${S.ds.id}/edit`, { method: "POST", body });
    renderDataView();
    return true;
  } catch (err) { toast(err.message, "err"); return false; }
}

async function doUndo() {
  try { S.ds = await api(`/api/datasets/${S.ds.id}/undo`, { method: "POST" }); renderDataView(); }
  catch (err) { toast(err.message, "err"); }
}

async function doRedo() {
  try { S.ds = await api(`/api/datasets/${S.ds.id}/redo`, { method: "POST" }); renderDataView(); }
  catch (err) { toast(err.message, "err"); }
}

function startCellEdit(td) {
  if (td.querySelector("input")) return;
  const old = td.textContent === "—" ? "" : td.textContent;
  td.innerHTML = `<input class="cell-input" value="${esc(old)}">`;
  const input = td.querySelector("input");
  input.focus(); input.select();
  let done = false;
  const commit = async (save) => {
    if (done) return; done = true;
    const v = input.value;
    if (save && v !== old) {
      const ok = await editSubmit({ action: "cell", row: +td.dataset.row, column: td.dataset.col, value: v });
      if (!ok) renderDataView();
    } else renderDataView();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") commit(true);
    else if (e.key === "Escape") commit(false);
  });
  input.addEventListener("blur", () => commit(true));
}

async function cycleType(col) {
  const order = ["quant", "cat", "ordinal"];
  const cur = S.ds.columns.find(c => c.name === col).dtype;
  const next = order[(order.indexOf(cur) + 1) % order.length] ?? "quant";
  await setType(col, next);
}

function variableMenu(chip) {
  const col = chip.dataset.col;
  const meta = S.ds.columns.find(c => c.name === col) || {};
  const pop = openPopover(chip, `<div class="pop-head">${esc(meta.label || col)} · 变量设置</div><div class="pop-list">${
    ["quant", "cat", "ordinal"].map(t => `<button class="pop-item" data-t="${t}">${vtBadge(t)} ${VT_LABEL[t]}</button>`).join("")
  }<button class="pop-item" data-act="label">✎ 编辑变量/值标签</button>
    <button class="pop-item" data-act="delcol" style="color:#e11d48">🗑 删除列</button></div>`);
  $$(".pop-item", pop).forEach(b => b.addEventListener("click", () => {
    closePopover();
    if (b.dataset.t) setType(col, b.dataset.t);
    else if (b.dataset.act === "label") openLabelModal(col);
    else if (confirm(`删除列「${col}」？该操作可撤销。`)) editSubmit({ action: "del_col", column: col });
  }));
}

async function setType(col, t) {
  try {
    await api(`/api/datasets/${S.ds.id}/columns`, { method: "PUT", body: { column: col, dtype: t } });
    S.ds = await api(`/api/datasets/${S.ds.id}?page=${S.dataPage}`);
    renderDataView();
    toast(`「${col}」→ ${VT_LABEL[t]}`, "ok");
  } catch (err) { toast(err.message, "err"); }
}

/* 变量标签 / 值标签编辑 */
function openLabelModal(col) {
  const meta = S.ds.columns.find(c => c.name === col) || {};
  const valsText = Object.entries(meta.vals || {}).map(([k, v]) => `${k}=${v}`).join("\n");
  openModal(`
    <div class="modal-head"><h3>标签 · ${esc(col)}</h3><button class="modal-close">×</button></div>
    <div class="modal-body">
      <div class="opt-row"><label class="field-label">变量标签（结果中更易读的名称）</label>
        <input id="lbVar" class="input" value="${esc(meta.label || "")}" placeholder="例:治疗疗效评级"></div>
      <div class="opt-row"><label class="field-label">值标签（每行一条，格式：原值=标签；sav 上传时自动读取）</label>
        <textarea id="lbVals" class="textarea" rows="6" placeholder="1=无效\n2=有效\n3=显效">${esc(valsText)}</textarea></div>
    </div>
    <div class="modal-foot">
      <button class="btn" id="lbCancel">取消</button>
      <button class="btn primary" id="lbSave">保存标签</button>
    </div>`);
  $("#lbCancel").addEventListener("click", closeModal);
  $("#lbSave").addEventListener("click", async () => {
    const vals = {};
    $("#lbVals").value.split("\n").forEach(line => {
      const i = line.indexOf("=");
      if (i > 0) { const k = line.slice(0, i).trim(), v = line.slice(i + 1).trim(); if (k) vals[k] = v; }
    });
    try {
      await api(`/api/datasets/${S.ds.id}/labels`, { method: "POST", body: { column: col, label: $("#lbVar").value, vals } });
      closeModal();
      toast("标签已保存", "ok");
      S.ds = await api(`/api/datasets/${S.ds.id}?page=${S.dataPage}&page_size=${S.pageSize}`);
      renderDataView();
    } catch (err) { toast(err.message, "err"); }
  });
}

/* 加权个案:选择权重列或取消加权 */
function weightMenu(anchor) {
  const quantCols = S.ds.columns.filter(c => c.dtype === "quant");
  const pop = openPopover(anchor, `<div class="pop-head">加权个案</div><div class="pop-list">
    <button class="pop-item ${S.ds.weightCol ? "" : "active"}" data-w="">⚖ 不加权</button>
    ${quantCols.map(c => `<button class="pop-item ${S.ds.weightCol === c.name ? "active" : ""}" data-w="${esc(c.name)}">⚖ ${esc(c.name)}</button>`).join("")
    || `<div style="padding:10px;font-size:12px;color:var(--ink-3)">暂无定量列可作为权重</div>`}
  </div>`);
  $$(".pop-item", pop).forEach(b => b.addEventListener("click", async () => {
    closePopover();
    try {
      await api(`/api/datasets/${S.ds.id}/weight`, { method: "POST", body: { column: b.dataset.w || null } });
      S.ds = await api(`/api/datasets/${S.ds.id}?page=${S.dataPage}&page_size=${S.pageSize}`);
      renderDataView();
      toast(b.dataset.w ? `已按「${b.dataset.w}」加权：描述统计/频数/卡方将使用加权计算` : "已取消加权", "ok");
    } catch (err) { toast(err.message, "err"); }
  }));
}

/* 数据集合并:纵向追加 / 横向关联,生成新数据集 */
function openMergeModal() {
  const others = S.datasets.filter(d => d.id !== S.ds.id);
  if (!others.length) { toast("只有一个数据集，先上传或载入另一个", "err"); return; }
  const colOpts = S.ds.columns.map(c => `<option value="${esc(c.name)}">${esc(c.name)}</option>`).join("");
  openModal(`
    <div class="modal-head"><h3>合并数据集 · ${esc(S.ds.name)}</h3><button class="modal-close">×</button></div>
    <div class="modal-body">
      <div class="opt-row"><label class="field-label">另一数据集</label>
        <select id="mgOther" class="select">${others.map(d => `<option value="${d.id}">${esc(d.name)}（${d.rows}×${d.cols}）</option>`).join("")}</select></div>
      <div class="opt-row"><label class="field-label">合并方式</label>
        <select id="mgHow" class="select">
          <option value="append">纵向追加（上下拼接，按列名对齐）</option>
          <option value="join">横向关联（按关键列左右连接）</option>
        </select></div>
      <div id="mgJoinBox" style="display:none">
        <div class="opt-row"><label class="field-label">关联键列（需两表同名）</label>
          <select id="mgOn" class="select">${colOpts}</select></div>
        <div class="opt-row"><label class="field-label">关联方式</label>
          <select id="mgJoin" class="select"><option value="inner">内联（取交集）</option><option value="left">左连（保留左表全部行）</option></select></div>
      </div>
      <p class="opt-tip">合并生成新数据集，原数据不变。</p>
    </div>
    <div class="modal-foot">
      <button class="btn" id="mgCancel">取消</button>
      <button class="btn primary" id="mgGo">合并</button>
    </div>`);
  $("#mgHow").addEventListener("change", () => {
    $("#mgJoinBox").style.display = $("#mgHow").value === "join" ? "" : "none";
  });
  $("#mgCancel").addEventListener("click", closeModal);
  $("#mgGo").addEventListener("click", async () => {
    try {
      const ds = await api(`/api/datasets/${S.ds.id}/merge`, { method: "POST", body: {
        other_id: $("#mgOther").value, how: $("#mgHow").value,
        on: $("#mgOn") ? $("#mgOn").value : "", join: $("#mgJoin") ? $("#mgJoin").value : "inner",
      } });
      closeModal();
      await loadDatasets();
      await selectDataset(ds.id, { silent: true });
      if (ds.evicted) toast(`数据集已达上限（30 个），最早的「${ds.evicted}」已被清理`, "err");
      toast(`已合并：${ds.rows} 行 × ${ds.cols} 列`, "ok");
    } catch (err) { toast(err.message, "err"); }
  });
}

/* 数据集导出:下载当前数据(工作台编辑、模型写回的派生列一并带出) */
function exportDsMenu(anchor) {
  const pop = openPopover(anchor, `<div class="pop-head">导出数据集 · ${esc(S.ds.name)}</div><div class="pop-list">
    <button class="pop-item" data-fmt="csv">CSV（UTF-8 带 BOM，Excel 可直接打开）</button>
    <button class="pop-item" data-fmt="xlsx">Excel 工作簿（.xlsx）</button></div>`);
  $$(".pop-item", pop).forEach(b => b.addEventListener("click", () => {
    closePopover();
    const a = document.createElement("a");
    a.href = `/api/datasets/${S.ds.id}/export?format=${b.dataset.fmt}`;
    a.download = `${S.ds.name}.${b.dataset.fmt}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    toast("已开始下载数据集", "ok");
  }));
}

/* 数据集重命名 */
function openRenameDsModal() {
  openModal(`
    <div class="modal-head"><h3>重命名数据集</h3><button class="modal-close">×</button></div>
    <div class="modal-body">
      <div class="opt-row"><label class="field-label">数据集名称</label>
        <input id="dsNameInput" class="input" value="${esc(S.ds.name)}" maxlength="80"></div>
    </div>
    <div class="modal-foot">
      <button class="btn" id="dsNameCancel">取消</button>
      <button class="btn primary" id="dsNameSave">保存</button>
    </div>`);
  $("#dsNameCancel").addEventListener("click", closeModal);
  $("#dsNameSave").addEventListener("click", async () => {
    const name = $("#dsNameInput").value.trim();
    if (!name) { toast("名称不能为空", "err"); return; }
    try {
      await api(`/api/datasets/${S.ds.id}/name`, { method: "PUT", body: { name } });
      closeModal();
      toast("已重命名", "ok");
      S.ds.name = name;
      await loadDatasets();
      renderDatasetBox();
      renderDataView();
    } catch (err) { toast(err.message, "err"); }
  });
  $("#dsNameInput").focus();
}

/* ---------------- 数据处理 ---------------- */
const PROCS = [
  { id: "fill_missing", name: "缺失值填充", icon: "♀" },
  { id: "drop_missing", name: "删除缺失行", icon: "⌫" },
  { id: "winsorize", name: "异常值缩尾", icon: "✂" },
  { id: "standardize", name: "标准化", icon: "±" },
  { id: "dummy", name: "哑变量", icon: "⑃" },
  { id: "recode_bins", name: "分箱重编码", icon: "▤" },
  { id: "compute", name: "计算变量", icon: "∑" },
  { id: "filter", name: "筛选行", icon: "搜寻" },
  { id: "sort", name: "排序", icon: "↕" },
  { id: "cast", name: "类型转换", icon: "⇄" },
];

function openProcessModal() {
  const ds = S.ds;
  const colOpts = ds.columns.map(c => `<option value="${esc(c.name)}">${esc(c.name)}</option>`).join("");
  const quantOpts = ds.columns.filter(c => c.dtype === "quant").map(c => `<option value="${esc(c.name)}">${esc(c.name)}</option>`).join("");
  const forms = {
    fill_missing: `<div class="opt-row"><label class="field-label">目标列（可多选）</label><select id="pfCols" class="select" multiple size="6">${colOpts}</select></div>
      <div class="opt-row"><label class="field-label">填充方式</label><select id="pfMethod" class="select">
        <option value="mean">均值(定量)</option><option value="median">中位数(定量)</option><option value="mode">众数</option>
        <option value="ffill">前一有效值</option><option value="constant">常数</option></select></div>
      <div class="opt-row"><label class="field-label">常数值（方式选常数时）</label><input id="pfValue" class="input" value="0"></div>`,
    drop_missing: `<div class="opt-row"><label class="field-label">检查列（留空=所有列）</label><select id="pfCols" class="select" multiple size="6">${colOpts}</select></div>
      <div class="opt-row"><label class="field-label">删除条件</label><select id="pfHow" class="select"><option value="any">任一列缺失</option><option value="all">全部列缺失</option></select></div>`,
    winsorize: `<div class="opt-row"><label class="field-label">定量列（可多选）</label><select id="pfCols" class="select" multiple size="6">${quantOpts}</select></div>
      <div class="opt-row"><label class="field-label">判定方法</label><select id="pfMethod" class="select"><option value="3sigma">3σ 法则</option><option value="iqr">四分位 IQR</option></select></div>`,
    standardize: `<div class="opt-row"><label class="field-label">定量列（可多选）</label><select id="pfCols" class="select" multiple size="6">${quantOpts}</select></div>
      <div class="opt-row"><label class="field-label">方法</label><select id="pfMethod" class="select">
        <option value="zscore">Z 标准化（均值0方差1）</option><option value="minmax">归一化 [0,1]</option>
        <option value="center">中心化</option><option value="robust">稳健标准化（中位数/IQR）</option></select></div>`,
    dummy: `<div class="opt-row"><label class="field-label">定类列</label><select id="pfCol" class="select">${colOpts}</select></div>`,
    recode_bins: `<div class="opt-row"><label class="field-label">定量列</label><select id="pfCol" class="select">${quantOpts}</select></div>
      <div class="opt-row"><label class="field-label">组数</label><input id="pfBins" class="input" type="number" value="3" min="2" max="10"></div>
      <div class="opt-row"><label class="field-label">新列名</label><input id="pfNew" class="input" value="分组"></div>`,
    compute: `<div class="opt-row"><label class="field-label">公式（可用列名与 + - * / ( )，例：月薪 * 1.2 + 工龄 * 100）</label>
      <textarea id="pfExpr" class="textarea" placeholder="月薪 / 1000"></textarea></div>
      <div class="opt-row"><label class="field-label">新列名</label><input id="pfNew" class="input" value="新变量"></div>`,
    filter: `<div class="opt-row"><label class="field-label">筛选表达式（例：年龄 > 30 and 性别 == '男'）</label>
      <textarea id="pfExpr" class="textarea" placeholder="总体满意度 >= 4"></textarea></div>`,
    sort: `<div class="opt-row"><label class="field-label">排序列（可多选，按顺序）</label><select id="pfCols" class="select" multiple size="5">${colOpts}</select></div>
      <div class="opt-row"><label class="field-label">方向</label><select id="pfAsc" class="select"><option value="1">升序</option><option value="0">降序</option></select></div>`,
    cast: `<div class="opt-row"><label class="field-label">目标列</label><select id="pfCol" class="select">${colOpts}</select></div>
      <div class="opt-row"><label class="field-label">转为</label><select id="pfTarget" class="select">
        <option value="quant">定量</option><option value="cat">定类</option><option value="ordinal">定级</option></select></div>`,
  };

  openModal(`
    <div class="modal-head"><h3>数据处理</h3><button class="modal-close">×</button></div>
    <div class="modal-body">
      <div class="proc-grid">${PROCS.map((p, i) => `<button class="proc-tile ${i === 0 ? "active" : ""}" data-op="${p.id}">${p.icon}<br>${p.name}</button>`).join("")}</div>
      <div id="procForm">${forms.fill_missing}</div>
    </div>
    <div class="modal-foot">
      <button class="btn" id="procCancel">取消</button>
      <button class="btn primary" id="procApply">应用到数据集</button>
    </div>`, { wide: true });

  let op = "fill_missing";
  $$(".proc-tile").forEach(t => t.addEventListener("click", () => {
    $$(".proc-tile").forEach(x => x.classList.remove("active"));
    t.classList.add("active");
    op = t.dataset.op;
    $("#procForm").innerHTML = forms[op] || "";
  }));
  $("#procCancel").addEventListener("click", closeModal);
  $("#procApply").addEventListener("click", async () => {
    const multi = [...($$("#pfCols")?.[0]?.selectedOptions || [])].map(o => o.value);
    const v = (id) => { const el = $(id); return el ? el.value : undefined; };
    const paramsMap = {
      fill_missing: { cols: multi, method: v("#pfMethod"), value: parseFloat(v("#pfValue")) || 0 },
      drop_missing: { cols: multi, how: v("#pfHow") },
      winsorize: { cols: multi, method: v("#pfMethod") },
      standardize: { cols: multi, method: v("#pfMethod") },
      dummy: { col: v("#pfCol"), drop_first: true },
      recode_bins: { col: v("#pfCol"), bins: parseInt(v("#pfBins")), new_name: v("#pfNew") },
      compute: { expr: v("#pfExpr"), new_name: v("#pfNew") },
      filter: { expr: v("#pfExpr") },
      sort: { cols: multi, ascending: [v("#pfAsc") === "1"] },
      cast: { col: v("#pfCol"), target: v("#pfTarget") },
    };
    try {
      const r = await api(`/api/datasets/${S.ds.id}/process`, { method: "POST", body: { op, params: paramsMap[op] } });
      closeModal();
      toast(r.message, "ok");
      S.ds = await api(`/api/datasets/${S.ds.id}?page=1`);
      S.dataPage = 1;
      renderDataView();
    } catch (err) { toast(err.message, "err"); }
  });
}

/* ---------------- 模型分析页 ---------------- */
async function viewModel(id) {
  if (!S.catalog) await loadCatalog();
  let found = null;
  for (const c of S.catalog.categories) {
    const hit = c.models.find(m => m.id === id);
    if (hit) { found = { ...hit, category: c.name }; break; }
  }
  if (!found) { location.hash = "#/"; return; }
  const prevModel = S.model;
  S.model = found;
  // 历史回放:恢复当时的槽位与选项(剔除已不存在的变量)
  const rp = S._replay && S._replay.modelId === id ? S._replay : null;
  S._replay = null;
  if (!S.ds) {
    const ds = await ensureDs();
    if (!ds) { location.hash = "#/"; toast("请先载入数据集"); return; }
  }
  renderTree($("#modelSearch").value);
  setTopbar(found.category, found.name, `
    ${S.ds ? `<span class="chip-note">数据集：${esc(S.ds.name)}</span>` : ""}
    <button class="btn" id="btnData">查看数据</button>`);
  $("#btnData").addEventListener("click", () => location.hash = `#/data/${S.ds.id}`);

  // 初始化槽位与选项(切换模型时重置,同模型内保留已选)
  if (rp) {
    S.slots = {};
    for (const sl of found.slots) S.slots[sl.key] = sl.multiple ? [] : "";
    Object.assign(S.slots, rp.slots || {});
    const valid = new Set(S.ds.columns.map(c => c.name));
    let dropped = [];
    for (const [k, v] of Object.entries(S.slots)) {
      const cols = Array.isArray(v) ? v : v ? [v] : [];
      const bad = cols.filter(c => !valid.has(c));
      if (bad.length) {
        dropped.push(...bad);
        S.slots[k] = Array.isArray(v) ? v.filter(c => valid.has(c)) : "";
      }
    }
    if (dropped.length) toast(`回放：变量「${dropped.join("、")}」已不在当前数据集中，已移除`, "err");
    S.options = { ...rp.options };
  } else if (!prevModel || prevModel.id !== found.id) {
    S.slots = {};
    for (const sl of found.slots) S.slots[sl.key] = sl.multiple ? [] : "";
    S.options = {};
    for (const o of found.options) S.options[o.key] = o.default;
  }
  for (const o of found.options) if (!(o.key in S.options)) S.options[o.key] = o.default;

  const accepted = (slot) => new Set(slot.types);
  const colInfo = Object.fromEntries(S.ds.columns.map(c => [c.name, c]));

  const slotHtml = found.slots.map(sl => {
    const val = S.slots[sl.key];
    const chips = (sl.multiple ? val : val ? [val] : []).map(c =>
      `<span class="var-chip">${vtBadge(colInfo[c]?.dtype || "any")}${esc(c)}<button class="x" data-slot="${sl.key}" data-col="${esc(c)}" title="移除">×</button></span>`).join("");
    const need = sl.required ? '<span class="req">*</span>' : "";
    const typesTip = sl.types.map(t => VT_LABEL[t]).join("/");
    return `<div class="opt-row">
      <label class="field-label">${esc(sl.label)} ${need}
        <span class="tip" title="可拖入类型:${typesTip}">${sl.multiple ? "可多选 · " : ""}${typesTip}</span></label>
      <div class="slot-box ${chips ? "filled" : ""}" data-slot="${sl.key}" tabindex="0" role="button" aria-label="选择${esc(sl.label)}">
        ${chips || `<span class="slot-hint">点击选择变量…</span>`}
        <button class="slot-add" title="添加变量">＋</button>
      </div>
      ${sl.tip ? `<div class="opt-tip">${esc(sl.tip)}</div>` : ""}
    </div>`;
  }).join("");

  const optHtml = found.options.map(o => {
    const vis = !o.showIf || S.options[o.showIf[0]] === o.showIf[1];
    if (!vis) return "";
    if (o.kind === "select") {
      const choices = (o.choices || []).map(c => `<option value="${esc(String(c.value))}" ${String(S.options[o.key]) === String(c.value) ? "selected" : ""}>${esc(c.label)}</option>`).join("");
      return `<div class="opt-row"><label class="field-label">${esc(o.label)}${o.tip ? `<span class="tip" title="${esc(o.tip)}">?</span>` : ""}</label><select class="select" data-opt="${o.key}">${choices}</select></div>`;
    }
    if (o.kind === "switch") {
      return `<div class="opt-row inline"><label class="field-label" style="margin:0">${esc(o.label)}</label>
        <label class="switch"><input type="checkbox" data-opt="${o.key}" ${S.options[o.key] ? "checked" : ""}><span class="track"></span></label></div>`;
    }
    if (o.kind === "textarea") {
      return `<div class="opt-row"><label class="field-label">${esc(o.label)}${o.tip ? `<span class="tip" title="${esc(o.tip)}">?</span>` : ""}</label>
        <textarea class="textarea" data-opt="${o.key}">${esc(S.options[o.key] ?? "")}</textarea></div>`;
    }
    const type = o.kind === "number" ? "number" : "text";
    const numAttr = o.kind === "number" ? ` step="${o.step ?? "any"}" ${o.min != null ? `min="${o.min}"` : ""} ${o.max != null ? `max="${o.max}"` : ""}` : "";
    return `<div class="opt-row"><label class="field-label">${esc(o.label)}${o.tip ? `<span class="tip" title="${esc(o.tip)}">?</span>` : ""}</label>
      <input class="input" type="${type}" data-opt="${o.key}" value="${esc(S.options[o.key] ?? "")}" ${numAttr}></div>`;
  }).join("");

  $("#view").innerHTML = `
  <div class="analysis">
    <div class="card config-card">
      <div class="model-brief">${esc(found.desc)}
        <div class="model-tags">${(found.tags || []).map(t => `<span>${esc(t)}</span>`).join("")}</div>
      </div>
      ${slotHtml}
      ${optHtml}
      <div class="run-bar">
        <button class="btn primary big" id="btnRun">▶ 运行分析</button>
      </div>
    </div>
    <div class="results" id="results">${S.result && S.result.model?.id === found.id ? "" : resultEmptyHtml(found)}</div>
  </div>`;

  // 主题切换等重渲染时,当前模型的既有结果直接重画,无需重新计算
  if (S.result && S.result.model?.id === found.id) renderResult();

  // 槽位事件
  $$(".slot-box").forEach(box => {
    const key = box.dataset.slot;
    box.addEventListener("click", (e) => {
      if (e.target.closest(".x")) return;
      openVarPicker(box, key);
    });
  });

  // 选项事件
  $$("[data-opt]").forEach(el => el.addEventListener("change", () => {
    const k = el.dataset.opt;
    S.options[k] = el.type === "checkbox" ? el.checked : el.value;
  }));

  $("#btnRun").addEventListener("click", runModel);
  if (rp) runModel();
}

function viewModelRefreshPanel() {
  const scroll = $("#view").scrollTop;
  viewModel(S.model.id).then(() => { $("#view").scrollTop = scroll; });
}

function openVarPicker(anchor, slotKey) {
  const spec = S.model.slots.find(s => s.key === slotKey);
  const chosen = new Set(spec.multiple ? S.slots[slotKey] : [S.slots[slotKey]].filter(Boolean));
  const usable = S.ds.columns.filter(c => spec.types.includes("any") || spec.types.includes(c.dtype));
  const items = usable.map(c => {
    const inAnother = !chosen.has(c.name) && Object.entries(S.slots).some(([k, v]) =>
      k !== slotKey && (S.model.slots.find(s => s.key === k)?.multiple ? v.includes(c.name) : v === c.name));
    const lv = [c.label, c.levels ? `${c.levels.length} 水平` : ""].filter(Boolean).join(" · ");
    return `<button class="pop-item" data-col="${esc(c.name)}" title="${esc(c.label || c.name)}">
      ${vtBadge(c.dtype)}<span>${esc(c.name)}</span><span class="pi-meta">${esc(lv)}</span></button>`;
  }).join("") || `<div style="padding:14px;font-size:12.5px;color:var(--ink-3)">当前数据集没有符合该槽位类型的变量，可到数据页修改变量类型。</div>`;
  const pop = openPopover(anchor, `<div class="pop-head">选择:${esc(spec.label)}<span style="font-weight:400">${spec.multiple ? "可多选" : "单选"}</span></div><div class="pop-list">${items}</div>`);
  $$(".pop-item", pop).forEach(b => b.addEventListener("click", () => {
    const col = b.dataset.col;
    if (spec.multiple) {
      const arr = S.slots[slotKey];
      const i = arr.indexOf(col);
      if (i >= 0) arr.splice(i, 1); else arr.push(col);
    } else {
      S.slots[slotKey] = S.slots[slotKey] === col ? "" : col;
    }
    closePopover();
    viewModelRefreshPanel();
  }));
}

function resultEmptyHtml(model) {
  return `<div class="result-empty card">
    <div class="re-icon"><svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M4 19h16M7 15l3.5-4.5 3 2.5L18 7"/></svg></div>
    <h4>在左侧配置变量，点「运行分析」</h4>
    <p>结果将在此展示：智能解读 → 结果表格 → 可视化图表。<br>当前模型：${esc(model.name)}</p>
  </div>`;
}

async function runModel() {
  if (S.running) return;
  S.running = true;
  const btn = $("#btnRun");
  btn.disabled = true;
  btn.innerHTML = `<span class="spin"></span> 计算中…`;
  $("#results").innerHTML = `<div class="card skeleton"><div class="sk-line" style="width:34%"></div><div class="sk-line" style="width:88%"></div><div class="sk-line" style="width:76%"></div><div class="sk-line" style="width:60%"></div><div class="sk-line" style="width:82%"></div></div>`;
  const t0 = performance.now();
  try {
    S.result = await api("/api/run", { method: "POST", body: { model: S.model.id, dataset: S.ds.id, slots: S.slots, options: S.options } });
    await reloadCurrentDsQuiet();
    renderResult();
    recordHistory(S.result);
    renderSideNav();
    toast(`分析完成(${((performance.now() - t0) / 1000).toFixed(2)}s)`, "ok");
  } catch (err) {
    $("#results").innerHTML = `<div class="card rcard error-card">
      <h3><span class="hbar" style="background:var(--rose)"></span>无法完成分析</h3>
      <div class="err-body">${esc(err.message)}</div></div>`;
  }
  btn.disabled = false;
  btn.innerHTML = "▶ 运行分析";
  S.running = false;
}

async function reloadCurrentDsQuiet() {
  if (!S.ds) return;
  try { S.ds = await api(`/api/datasets/${S.ds.id}?page=${S.dataPage}`); renderDatasetBox(); } catch { /* ignore */ }
}

/* ---------------- 综合分析报告 ---------------- */
function recordHistory(result) {
  const slotsText = (S.model?.slots || [])
    .filter(s => s.required)
    .map(s => {
      const v = S.slots[s.key];
      const val = Array.isArray(v) ? v.join("、") : v;
      return `${s.label}:${val || "—"}`;
    }).join(";");
  S.history.unshift({
    time: new Date(),
    modelId: S.model?.id || "",
    modelName: S.model?.name || result?.model?.name || "分析",
    modelCat: result?.model?.category || S.model?.category || "",
    dsName: S.ds?.name || "数据集",
    dsId: S.ds?.id || "",
    slots: JSON.parse(JSON.stringify(S.slots || {})),
    options: JSON.parse(JSON.stringify(S.options || {})),
    slotsText,
    result,
  });
  if (S.history.length > 30) S.history.length = 30;
}

function renderSideNav() {
  const badge = $("#historyCount");
  if (badge) badge.textContent = S.history.length;
  const nav = $(".side-nav-item");
  if (nav) nav.classList.toggle("active", location.hash === "#/report");
}

function viewReport() {
  S.model = null;
  chartObservers.forEach(o => o.disconnect()); chartObservers = [];
  setTopbar("", "综合分析报告");
  renderTree($("#modelSearch").value);
  const v = $("#view");
  if (!S.history.length) {
    v.innerHTML = `<div class="card report-empty"><h3>暂无分析记录</h3>
      <p>运行任意模型后，本次会话的分析结果会自动汇总到这里，并可一键导出为可打印的 HTML 报告。</p>
      <a class="btn primary" href="#/">去运行第一个分析 →</a>
      <button class="btn" id="btnPlans" style="margin-left:8px">📋 查看分析方案</button></div>`;
    $("#btnPlans").addEventListener("click", openPlanDrawer);
    return;
  }
  const html = S.history.map((e, i) => `
    <section class="report-item card" id="rp${i}">
      <div class="report-item-head">
        <span class="report-no">${i + 1}</span>
        <div>
          <h3>${esc(e.modelName)}</h3>
          <div class="report-meta">${esc(e.modelCat)} · 数据集「${esc(e.dsName)}」 · ${e.time.toLocaleString("zh-CN")}</div>
          ${e.slotsText ? `<div class="report-slots">${esc(e.slotsText)}</div>` : ""}
        </div>
        <span style="flex:1"></span>
        ${e.modelId ? `<button class="btn ghost" data-replay="${i}" title="恢复当时的模型、数据集与参数并重新计算">↻ 重新运行</button>` : ""}
      </div>
      <div class="report-body">${resultFlowHtml(e.result, "all", `rp${i}-`)}</div>
    </section>`).join("");
  v.innerHTML = `
    <div class="report-top card">
      <div>
        <h2 style="margin:0 0 4px">本次会话共 ${S.history.length} 项分析</h2>
        <div class="report-meta">生成时间 ${new Date().toLocaleString("zh-CN")} · 导出后可用浏览器打印为 PDF</div>
      </div>
      <div style="display:flex;gap:8px">
        <button class="btn" id="btnPlanSave" title="把本次会话的分析存为可复用的分析方案">💾 存为方案</button>
        <button class="btn" id="btnPlans" title="管理已保存的分析方案">📋 方案</button>
        <button class="btn primary" id="btnReportExport">⬇ HTML</button>
        <button class="btn" id="btnReportDocx">⬇ Word</button>
        <button class="btn" id="btnReportPdf">⬇ PDF</button>
        <button class="btn ghost" id="btnReportClear">清空记录</button>
      </div>
    </div>
    <nav class="report-toc card">
      <b>目录</b>
      <ol>${S.history.map((e, i) => `<li><a href="#rp${i}">${esc(e.modelName)}<span class="toc-meta">${esc(e.dsName)}</span></a></li>`).join("")}</ol>
    </nav>
    ${html}`;
  $$("[data-replay]").forEach(b => b.addEventListener("click", () => replayHistory(+b.dataset.replay)));
  // 目录锚点不能真改 hash:应用是 hash 路由,#rp0 会被当作未知路由回退到首页
  $$(".report-toc a").forEach(a => a.addEventListener("click", ev => {
    ev.preventDefault();
    document.getElementById(a.getAttribute("href").slice(1))
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  }));
  $("#btnPlanSave").addEventListener("click", openPlanSaveModal);
  $("#btnPlans").addEventListener("click", openPlanDrawer);
  $("#btnReportExport").addEventListener("click", exportReportHtml);
  $("#btnReportDocx").addEventListener("click", () => exportReportFile("docx"));
  $("#btnReportPdf").addEventListener("click", () => exportReportFile("pdf"));
  $("#btnReportClear").addEventListener("click", () => {
    S.history = [];
    toast("已清空分析记录");
    viewReport();
  });
  S.history.forEach((e, i) => e.result.charts.forEach((c, j) => {
    const box = document.getElementById(`chart-rp${i}-${j}`);
    if (box) initChartIn(box, c);
  }));
}

/* 历史回放:切换到原数据集,恢复参数后自动重新运行(方案步骤共用 replayStep) */
async function replayHistory(i) {
  const e = S.history[i];
  if (!e?.modelId) return;
  await replayStep({ model: e.modelId, dsId: e.dsId, slots: e.slots, options: e.options });
}

async function replayStep(step) {
  if (!step?.model) return;
  if (!S.datasets.some(d => d.id === step.dsId)) { toast("原数据集已被删除，无法回放", "err"); return; }
  if (!S.ds || S.ds.id !== step.dsId) await selectDataset(step.dsId, { silent: true });
  S._replay = { modelId: step.model, slots: step.slots || {}, options: step.options || {} };
  const target = `#/m/${step.model}`;
  if (location.hash === target) route();
  else location.hash = target;
}

/* 收集报告载荷:解读/表格 + 已渲染图表的 PNG;Word 与 PDF 共用 */
function collectReport() {
  const dark = document.documentElement.dataset.theme === "dark";
  return {
    title: "数据分析报告",
    generated: `生成时间：${new Date().toLocaleString("zh-CN")} · 共 ${S.history.length} 项分析 · 由 OpenSPSS 生成`,
    dark,
    items: S.history.map((e, i) => ({
      title: `${i + 1}. ${e.modelName}`,
      meta: [e.modelCat, `数据集「${e.dsName}」`, e.time.toLocaleString("zh-CN")].filter(Boolean).join(" · "),
      slots: e.slotsText || "",
      paragraphs: [...(e.result.interpretation || []), ...(e.result.notes || [])],
      tables: e.result.tables.map(t => ({ title: t.title, headers: t.headers, rows: t.rows, note: t.note || "" })),
      charts: e.result.charts.map((c, j) => {
        const box = document.getElementById(`chart-rp${i}-${j}`);
        const inst = box && echarts.getInstanceByDom(box);
        const png = inst ? inst.getDataURL({ pixelRatio: 2, backgroundColor: dark ? "#171D36" : "#fff" }) : "";
        return { title: c.title, png };
      }).filter(c => c.png),
    })),
  };
}

async function exportReportFile(kind) {
  const payload = collectReport();
  if (!payload.items.length) { toast("暂无分析记录", "err"); return; }
  toast(`正在生成 ${kind.toUpperCase()}…`);
  try {
    const res = await fetch(`/api/export/report/${kind}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    if (!res.ok) throw new Error(`生成失败(HTTP ${res.status})`);
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `OpenSPSS分析报告_${new Date().toISOString().slice(0, 10)}.${kind}`;
    a.click();
    URL.revokeObjectURL(a.href);
    toast(`已导出 ${kind.toUpperCase()}`, "ok");
  } catch (err) { toast(err.message, "err"); }
}

function exportReportHtml() {
  const items = S.history.map((e, i) => {
    const chartImgs = e.result.charts.map((c, j) => {
      const box = document.getElementById(`chart-rp${i}-${j}`);
      const inst = box && echarts.getInstanceByDom(box);
      const url = inst ? inst.getDataURL({ pixelRatio: 2, backgroundColor: "#fff" }) : "";
      return `<h4>${esc(c.title)}</h4>${url ? `<img class="rp-chart" src="${url}" alt="${esc(c.title)}">` : "<p class='rp-muted'>(图表渲染失败)</p>"}`;
    }).join("");
    const tables = e.result.tables.map(t => `
      <h4>${esc(t.title)}</h4>
      <table class="rp-table"><thead><tr>${t.headers.map(h => `<th>${esc(h)}</th>`).join("")}</tr></thead>
      <tbody>${t.rows.map(r => `<tr>${r.map(cell => `<td>${pCell(String(cell ?? ""))}</td>`).join("")}</tr>`).join("")}</tbody></table>
      ${t.note ? `<div class="rp-note">${esc(t.note)}</div>` : ""}`).join("");
    return `<section class="rp-item">
      <h2>${i + 1}. ${esc(e.modelName)}</h2>
      <div class="rp-meta">${esc(e.modelCat)} · 数据集「${esc(e.dsName)}」 · ${e.time.toLocaleString("zh-CN")}</div>
      ${e.slotsText ? `<div class="rp-slots">${esc(e.slotsText)}</div>` : ""}
      ${e.result.interpretation.length ? `<div class="rp-interp">${e.result.interpretation.map(p => `<p>${md(p)}</p>`).join("")}</div>` : ""}
      ${e.result.notes?.length ? `<div class="rp-interp">${e.result.notes.map(p => `<p>${md(p)}</p>`).join("")}</div>` : ""}
      ${tables}${chartImgs}
      ${e.result.code ? `<h4>可复现代码</h4><pre class="rp-code">${esc(e.result.code)}</pre>` : ""}
    </section>`;
  }).join("");
  const dark = document.documentElement.dataset.theme === "dark";
  const cHi = dark ? "#93A0FF" : "#4653E8", cMut = dark ? "#8B96B5" : "#64748b",
        cLine = dark ? "#303A5E" : "#e2e8f0", cHead = dark ? "#1C2340" : "#f1f5f9",
        cStripe = dark ? "#151B33" : "#fafbfe", cInterp = dark ? "#232B52" : "#eef2f7",
        cSlot = dark ? "#1C2340" : "#f1f5f9";
  const doc = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>OpenSPSS 分析报告</title><style>
body{font-family:"Microsoft YaHei",system-ui,sans-serif;color:${dark ? "#DEE4F5" : "#1e293b"};background:${dark ? "#10152B" : "#fff"};max-width:960px;margin:0 auto;padding:32px 24px;line-height:1.65}
.rp-cover{border-bottom:3px solid ${cHi};text-align:center;padding:48px 0 24px;margin-bottom:24px}
.rp-cover h1{margin:0 0 8px;font-size:26px;letter-spacing:2px}
.rp-meta{color:${cMut};font-size:12.5px;margin:2px 0}
.rp-slots{background:${cSlot};border-radius:8px;padding:8px 12px;font-size:12.5px;color:#475569;margin:8px 0}
.rp-toc{background:${cStripe};border:1px solid ${cLine};border-radius:10px;padding:12px 24px;margin-bottom:28px;font-size:13.5px}
.rp-toc a{color:${cHi};text-decoration:none}
.rp-item{margin-bottom:36px}
.rp-item h2{border-left:4px solid ${cHi};padding-left:10px;font-size:19px}
.rp-item h4{margin:18px 0 6px;font-size:14.5px;color:#334155}
.rp-interp{background:${cInterp};border-radius:10px;padding:10px 14px;margin:10px 0}
.rp-table{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0}
.rp-table th{background:${cHead};text-align:left}
.rp-table th,.rp-table td{border:1px solid ${cLine};padding:5px 9px}
.rp-table tr:nth-child(even) td{background:${cStripe}}
.rp-chart{max-width:100%;border:1px solid ${cLine};border-radius:8px;margin:4px 0 12px}
.rp-code{background:#0f172a;color:#e2e8f0;border-radius:8px;padding:12px 14px;font-family:Consolas,monospace;font-size:11.5px;line-height:1.5;overflow-x:auto;white-space:pre}
.rp-note{color:${cMut};font-size:12px}
footer{border-top:1px solid ${cLine};margin-top:40px;padding-top:12px;color:${cMut};font-size:12px;text-align:center}
@media print{.rp-item{page-break-inside:avoid}body{padding:0}}
</style></head><body>
<div class="rp-cover"><h1>数据分析报告</h1>
<div class="rp-meta">由 OpenSPSS 开源统计分析平台生成</div>
<div class="rp-meta">导出时间:${new Date().toLocaleString("zh-CN")} · 共 ${S.history.length} 项分析</div></div>
<div class="rp-toc"><b>目录</b><ol>${S.history.map((e, i) => `<li><a href="#i${i}">${esc(e.modelName)} — ${esc(e.dsName)}</a></li>`).join("")}</ol></div>
${items}
<footer>OpenSPSS · MIT License · 本报告由浏览器打印功能可另存为 PDF</footer>
</body></html>`;
  const blob = new Blob([doc], { type: "text/html;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `OpenSPSS分析报告_${new Date().toISOString().slice(0, 10)}.html`;
  a.click();
  URL.revokeObjectURL(a.href);
  toast("报告已导出", "ok");
}

/* ---------------- 分析方案:保存 / 管理 / 批量重放 ---------------- */
/* 方案只存「怎么算」(模型 + 数据集引用 + 槽位/选项),结果由运行现场生成,不计入会话报告 */
function stepsFromHistory() {
  return [...S.history].reverse()
    .filter(e => e.modelId)
    .map(e => ({ model: e.modelId, modelName: e.modelName, dsId: e.dsId, dsName: e.dsName,
                 slots: e.slots, options: e.options, slotsText: e.slotsText || "" }));
}

function openPlanSaveModal() {
  const steps = stepsFromHistory();
  if (!steps.length) { toast("暂无分析记录，先运行分析再保存方案", "err"); return; }
  const defName = S.ds ? `${S.ds.name}·方案` : `分析方案 ${new Date().toISOString().slice(0, 10)}`;
  openModal(`
    <div class="modal-head"><h3>存为分析方案</h3><button class="modal-close">×</button></div>
    <div class="modal-body">
      <div class="opt-row"><label class="field-label">方案名称</label>
        <input id="planNameInput" class="input" value="${esc(defName)}" maxlength="60"></div>
      <p class="opt-tip">将保存本次会话的 ${steps.length} 项分析（模型、数据集与参数），可随时一键重放；不保存结果本身。</p>
    </div>
    <div class="modal-foot">
      <button class="btn" id="planSaveCancel">取消</button>
      <button class="btn primary" id="planSaveGo">保存方案</button>
    </div>`);
  $("#planSaveCancel").addEventListener("click", closeModal);
  $("#planSaveGo").addEventListener("click", async () => {
    try {
      const plan = await api("/api/plans", { method: "POST", body: { name: $("#planNameInput").value, steps } });
      closeModal();
      toast(`已保存方案「${plan.name}」（${plan.steps.length} 步）`, "ok");
      location.hash = `#/plan/${plan.id}`;
    } catch (err) { toast(err.message, "err"); }
  });
  $("#planNameInput").focus();
}

async function openPlanDrawer() {
  let plans = [];
  try { plans = (await api("/api/plans")).plans; }
  catch (err) { toast(err.message, "err"); return; }
  openModal(`
    <div class="modal-head"><h3>分析方案</h3><button class="modal-close">×</button></div>
    <div class="modal-body">
      ${plans.map(p => `
        <div class="plan-row" data-id="${p.id}">
          <div class="plan-row-main">
            <b>${esc(p.name)}</b>
            <span class="plan-row-meta">${p.steps.length} 步 · 更新于 ${new Date(p.updated * 1000).toLocaleString("zh-CN")}</span>
          </div>
          <button class="btn ghost" data-open="${p.id}">打开</button>
          <button class="btn ghost" data-ren="${p.id}" data-name="${esc(p.name)}" title="重命名">✎</button>
          <button class="btn ghost" data-delplan="${p.id}" title="删除" style="color:#e11d48">🗑</button>
        </div>`).join("") || `<p style="color:var(--ink-3);font-size:13px">还没有方案。在综合分析报告页点「💾 存为方案」即可保存。</p>`}
    </div>
    <div class="modal-foot"><button class="btn" id="planClose">关闭</button></div>`);
  $("#planClose").addEventListener("click", closeModal);
  $$(".plan-row [data-open]").forEach(b => b.addEventListener("click", () => { closeModal(); location.hash = `#/plan/${b.dataset.open}`; }));
  $$(".plan-row [data-ren]").forEach(b => b.addEventListener("click", () => renamePlanModal(b.dataset.ren, b.dataset.name, openPlanDrawer)));
  $$(".plan-row [data-delplan]").forEach(b => b.addEventListener("click", async () => {
    if (!confirm("删除该分析方案？该操作不可恢复。")) return;
    try {
      await api(`/api/plans/${b.dataset.delplan}`, { method: "DELETE" });
      toast("方案已删除", "ok");
      closeModal();
      openPlanDrawer();
    } catch (err) { toast(err.message, "err"); }
  }));
}

function renamePlanModal(pid, name, after) {
  openModal(`
    <div class="modal-head"><h3>重命名方案</h3><button class="modal-close">×</button></div>
    <div class="modal-body">
      <div class="opt-row"><label class="field-label">方案名称</label>
        <input id="planRenInput" class="input" value="${esc(name || "")}" maxlength="60"></div>
    </div>
    <div class="modal-foot">
      <button class="btn" id="planRenCancel">取消</button>
      <button class="btn primary" id="planRenSave">保存</button>
    </div>`);
  $("#planRenCancel").addEventListener("click", closeModal);
  $("#planRenSave").addEventListener("click", async () => {
    try {
      await api(`/api/plans/${pid}`, { method: "PUT", body: { name: $("#planRenInput").value } });
      closeModal();
      toast("已重命名", "ok");
      if (after) after();
    } catch (err) { toast(err.message, "err"); }
  });
  $("#planRenInput").focus();
}

/* 方案页:展示步骤清单;运行后逐步渲染结果/失败/跳过状态 */
async function viewPlan(pid) {
  S.model = null;
  chartObservers.forEach(o => o.disconnect()); chartObservers = [];
  renderTree($("#modelSearch").value);
  const v = $("#view");
  let plan;
  try { plan = await api(`/api/plans/${pid}`); }
  catch { toast("方案不存在或已被删除", "err"); location.hash = "#/"; return; }
  setTopbar("分析方案", plan.name, `
    <span class="chip-note">${plan.steps.length} 个步骤 · 更新于 ${new Date(plan.updated * 1000).toLocaleString("zh-CN")}</span>
    <button class="btn primary" id="btnPlanRun" ${plan.steps.length ? "" : "disabled"}>▶ 运行全部</button>
    <button class="btn" id="btnPlanAppend" ${S.history.length ? "" : "disabled"} title="把本次会话的分析追加到方案末尾">＋ 追加会话分析</button>
    <button class="btn" id="btnPlanRename">✎ 重命名</button>`);
  $("#btnPlanRun").addEventListener("click", () => runPlan(pid));
  $("#btnPlanAppend")?.addEventListener("click", () => appendSessionToPlan(pid));
  $("#btnPlanRename").addEventListener("click", () => renamePlanModal(pid, plan.name, () => viewPlan(pid)));

  const run = S.planRun && S.planRun.planId === pid ? S.planRun : null;
  const stepsHtml = plan.steps.map((s, i) => planStepHtml(s, i, run?.outcomes?.[i])).join("")
    || `<div class="card report-empty"><h3>空方案</h3><p>该方案还没有步骤，可运行分析后点「＋ 追加会话分析」。</p></div>`;
  v.innerHTML = `
  <div class="page-pad">
    ${run ? `<div class="card plan-summary ${run.outcomes.some(o => o.status !== "ok") ? "warn" : ""}">${planRunSummary(run)}</div>` : `
      <div class="card report-empty">
        <h3>方案共 ${plan.steps.length} 个分析步骤</h3>
        <p>点「▶ 运行全部」按顺序在各自的数据集上重放所有步骤；数据集已被删除的步骤会标记跳过，可改绑到当前数据集。单步也可「↻ 重新运行」。</p>
      </div>`}
    ${stepsHtml}
  </div>`;
  if (run) {
    run.outcomes.forEach((o, i) => o.result?.charts?.forEach((c, j) => {
      const box = document.getElementById(`chart-po${i}-${j}`);
      if (box) initChartIn(box, c);
    }));
  }
  $$("[data-step-replay]").forEach(b => b.addEventListener("click", () => replayStep(plan.steps[+b.dataset.stepReplay])));
  $$("[data-step-del]").forEach(b => b.addEventListener("click", async () => {
    const i = +b.dataset.stepDel;
    if (!confirm(`从方案中移除第 ${i + 1} 步「${plan.steps[i].modelName}」？`)) return;
    try {
      await api(`/api/plans/${pid}`, { method: "PUT", body: { steps: plan.steps.filter((_, k) => k !== i) } });
      S.planRun = null;
      toast("已移除该步骤", "ok");
      viewPlan(pid);
    } catch (err) { toast(err.message, "err"); }
  }));
  $$("[data-rebind]").forEach(b => b.addEventListener("click", async () => {
    if (!S.ds) { toast("请先在首页或数据页选择一个数据集", "err"); return; }
    const i = +b.dataset.rebind;
    const steps = plan.steps.map((s, k) => k === i ? { ...s, dsId: S.ds.id, dsName: S.ds.name } : s);
    try {
      await api(`/api/plans/${pid}`, { method: "PUT", body: { steps } });
      S.planRun = null;
      toast(`第 ${i + 1} 步已改绑到「${S.ds.name}」`, "ok");
      viewPlan(pid);
    } catch (err) { toast(err.message, "err"); }
  }));
}

function planStepHtml(s, i, outcome) {
  const meta = [s.dsName ? `数据集「${esc(s.dsName)}」` : "", s.slotsText ? esc(s.slotsText) : ""].filter(Boolean).join(" · ");
  let status = "", body = "";
  if (outcome?.status === "ok") {
    status = `<span class="plan-badge ok">✓ 成功</span>`;
    body = `<div class="report-body">${resultFlowHtml(outcome.result, "all", `po${i}-`)}</div>`;
  } else if (outcome?.status === "skipped") {
    status = `<span class="plan-badge skip">⊘ 跳过</span>`;
    body = `<div class="card rcard plan-err"><div class="err-body">${esc(outcome.error || "")}</div>
      ${s.dsId ? `<button class="btn" data-rebind="${i}" ${S.ds ? "" : "disabled"} title="把该步骤的数据集替换为当前选中的数据集">改绑到${S.ds ? `「${esc(S.ds.name)}」` : "当前数据集"}</button>` : ""}</div>`;
  } else if (outcome?.status === "error") {
    status = `<span class="plan-badge err">✕ 失败</span>`;
    body = `<div class="card rcard plan-err"><div class="err-body">${esc(outcome.error || "")}</div></div>`;
  }
  return `<section class="report-item card" id="ps${i}">
    <div class="report-item-head">
      <span class="report-no">${i + 1}</span>
      <div>
        <h3>${esc(s.modelName || s.model)}</h3>
        <div class="report-meta">${meta}</div>
      </div>
      <span style="flex:1"></span>
      ${status}
      <button class="btn ghost" data-step-replay="${i}" title="切到原数据集，恢复参数并重新计算">↻ 重新运行</button>
      <button class="btn ghost" data-step-del="${i}" title="从方案中移除该步骤">移除</button>
    </div>
    ${body}
  </section>`;
}

function planRunSummary(run) {
  const ok = run.outcomes.filter(o => o.status === "ok").length;
  const skip = run.outcomes.filter(o => o.status === "skipped").length;
  const err = run.outcomes.filter(o => o.status === "error").length;
  return `本次运行：共 ${run.outcomes.length} 步 · <b>${ok}</b> 项成功` +
    (skip ? ` · ${skip} 项跳过` : "") + (err ? ` · ${err} 项失败` : "") +
    `（结果仅在此页展示，不计入会话综合报告）`;
}

async function runPlan(pid) {
  const btn = $("#btnPlanRun");
  if (!btn || btn.disabled) return;
  btn.disabled = true;
  btn.innerHTML = `<span class="spin"></span> 运行中…`;
  try {
    const r = await api(`/api/plans/${pid}/run`, { method: "POST" });
    S.planRun = { planId: pid, outcomes: r.outcomes };
    const skip = r.outcomes.filter(o => o.status === "skipped").length;
    const err = r.outcomes.filter(o => o.status === "error").length;
    const ok = r.outcomes.length - skip - err;
    toast(`方案运行完成：${ok} 项成功${skip ? `，${skip} 项跳过` : ""}${err ? `，${err} 项失败` : ""}`, ok === r.outcomes.length ? "ok" : "err");
  } catch (err) { toast(err.message, "err"); }
  viewPlan(pid);
}

async function appendSessionToPlan(pid) {
  const extra = stepsFromHistory();
  if (!extra.length) { toast("本次会话还没有分析记录", "err"); return; }
  try {
    const plan = await api(`/api/plans/${pid}`);
    await api(`/api/plans/${pid}`, { method: "PUT", body: { steps: [...plan.steps, ...extra] } });
    S.planRun = null;
    toast(`已追加 ${extra.length} 项会话分析到方案`, "ok");
    viewPlan(pid);
  } catch (err) { toast(err.message, "err"); }
}

/* ---------------- 主题 ---------------- */
function applyTheme(t) {
  document.documentElement.dataset.theme = t === "dark" ? "dark" : "";
  localStorage.setItem("ospss.theme", document.documentElement.dataset.theme);
  const btn = $("#themeToggle");
  if (btn) btn.textContent = document.documentElement.dataset.theme === "dark" ? "☀️ 明色" : "🌙 暗色";
  route();      // 重渲染当前视图,让 ECharts 按新主题重绘
}

/* ---------------- 启动 ---------------- */
async function boot() {
  renderSideNav();
  try {
    const [health] = await Promise.all([api("/api/health"), loadCatalog(), loadDatasets()]);
    if (health?.frozen) {
      const ex = $("#exitApp");
      if (ex) {
        ex.style.display = "";
        ex.addEventListener("click", (e) => {
          e.preventDefault();
          if (!confirm("停止本地 OpenSPSS 服务并退出？")) return;
          api("/api/shutdown", { method: "POST" }).finally(() => {
            document.body.innerHTML = `<div style="display:flex;height:100vh;align-items:center;justify-content:center;color:#64748b;font-size:14px">OpenSPSS 已停止,可关闭此窗口。</div>`;
          });
        });
      }
    }
  } catch (err) {
    $("#view").innerHTML = `<div class="result-empty card" style="margin:40px"><h4>无法连接后端服务</h4><p>${esc(err.message)}</p></div>`;
    return;
  }
  $("#modelSearch").addEventListener("input", (e) => renderTree(e.target.value));
  const tt = $("#themeToggle");
  tt.textContent = document.documentElement.dataset.theme === "dark" ? "☀️ 明色" : "🌙 暗色";
  tt.addEventListener("click", (e) => {
    e.preventDefault();
    applyTheme(document.documentElement.dataset.theme === "dark" ? "" : "dark");
  });

  // 移除已选变量(全局事件委托,替代每次渲染的临时监听)
  $("#view").addEventListener("click", (e) => {
    const x = e.target.closest(".x[data-slot]");
    if (!x) return;
    e.stopPropagation();
    const { slot, col } = x.dataset;
    if (S.model?.slots.find(s => s.key === slot)?.multiple) S.slots[slot] = S.slots[slot].filter(c => c !== col);
    else S.slots[slot] = "";
    viewModelRefreshPanel();
  });
  // 数据页快捷键:Ctrl+Z / Ctrl+Y 撤销与重做
  document.addEventListener("keydown", (e) => {
    if (!location.hash.startsWith("#/data/") || !(e.ctrlKey || e.metaKey)) return;
    if (e.target.closest && e.target.closest("input,textarea,select")) return;
    if (e.key === "z" || e.key === "Z") { e.preventDefault(); doUndo(); }
    if (e.key === "y" || e.key === "Y") { e.preventDefault(); doRedo(); }
  });
  window.addEventListener("hashchange", route);
  route();
}
boot();
