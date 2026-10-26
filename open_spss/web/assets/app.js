/* OpenSPSS 前端应用 */
"use strict";

/* ---------------- 状态与工具 ---------------- */
const S = {
  catalog: null,            // {categories, total}
  datasets: [], samples: [],
  ds: null,                 // 当前数据集详情
  model: null,              // 当前模型 spec
  slots: {}, options: {},
  result: null,
  running: false,
  dataPage: 1,
  history: [],              // 本次会话的分析记录(供综合报告)
};
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const VT_LABEL = { quant: "定量", cat: "定类", ordinal: "定级", text: "文本", any: "任意" };
const vtBadge = (t) => `<span class="vt ${t}">${VT_LABEL[t] || t}</span>`;

function toast(msg, kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; el.style.transition = "opacity .3s"; setTimeout(() => el.remove(), 320); }, 3200);
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : undefined,
    ...opts,
    body: opts.body && !(opts.body instanceof FormData) ? JSON.stringify(opts.body) : opts.body,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try { detail = (await res.json()).detail || detail; } catch { /* ignore */ }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

/* 极简 markdown:**bold** 与 `code` */
function md(text) {
  return esc(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

/* p 值单元格:色点 + 星号 */
function pCell(text) {
  if (!/[pP<]|0\.\d|<0\.001/.test(text)) return esc(text);
  let dot = "sig0";
  if (text.includes("***")) dot = "sig1";
  else if (text.includes("**")) dot = "sig2";
  else if (text.includes("*")) dot = "sig3";
  return `<span class="pv"><span class="dot ${dot}"></span>${esc(text)}</span>`;
}

/* ---------------- 弹窗 ---------------- */
function openModal(html, { wide = false } = {}) {
  const root = $("#modal-root");
  root.innerHTML = `<div class="modal-mask"><div class="modal ${wide ? "wide" : ""}" role="dialog" aria-modal="true">${html}</div></div>`;
  const mask = $(".modal-mask", root);
  mask.addEventListener("mousedown", (e) => { if (e.target === mask) closeModal(); });
  $(".modal-close", root)?.addEventListener("click", closeModal);
  return mask;
}
function closeModal() { $("#modal-root").innerHTML = ""; }

document.addEventListener("mousedown", (e) => {
  const pop = $("#popover-root .popover");
  if (pop && !(e.target.closest && e.target.closest(".slot-box")) && !pop.contains(e.target)) closePopover();
});
function closePopover() { $("#popover-root").innerHTML = ""; }
function openPopover(anchor, html) {
  const root = $("#popover-root");
  root.innerHTML = `<div class="popover">${html}</div>`;
  const pop = $(".popover", root);
  const r = anchor.getBoundingClientRect();
  pop.style.top = Math.min(r.bottom + 6, innerHeight - 340) + "px";
  pop.style.left = Math.min(r.left, innerWidth - 286) + "px";
  return pop;
}

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
  if (!silent) toast(`已切换数据集:${S.ds.name}`, "ok");
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
  $("#modelCount").textContent = `${shown || S.catalog.total} 个模型`;
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
        ${d.source === "sample" ? "" : `<span class="del" data-del="${d.id}" title="删除">×</span>`}</button>`).join("")
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
  await api(`/api/datasets/${id}`, { method: "DELETE" });
  toast("数据集已删除");
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
        <h2>把数据放进来,<br><span class="grad">剩下的交给统计。</span></h2>
        <p>OpenSPSS 内置 <b>${S.catalog?.total ?? "…"} 个分析模型</b>:从描述统计、假设检验到回归建模、机器学习与综合评价。选择模型、拖入变量、一键得到结果与智能解读。</p>
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
      <h4>拖拽文件到这里,或点击选择</h4>
      <p>支持 csv / xlsx / xls / sav(SPSS) / tsv / json / parquet,60MB 以内</p>
      <input type="file" id="fileInput" hidden accept=".csv,.xlsx,.xls,.sav,.tsv,.json,.parquet">
    </div>

    <div class="sec-title">示例数据,即点即用</div>
    <div class="sample-grid">${samples}</div>

    <div class="sec-title">我的数据集</div>
    <div class="ds-list">${
      S.datasets.map(d => `
        <div class="ds-row" data-id="${d.id}" tabindex="0" role="button">
          <div><div class="dsr-name">${esc(d.name)} ${d.source === "sample" ? `<span class="badge-sample">示例</span>` : ""}</div>
          <div class="dsr-meta">${d.rows} 行 × ${d.cols} 列</div></div>
          <svg class="arrow" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 6l6 6-6 6"/></svg>
        </div>`).join("") || `<p style="color:var(--ink-3);font-size:13px">暂无,载入示例或上传文件后出现在这里。</p>`
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
    c.style.opacity = ".6";
    try {
      const ds = await api(`/api/datasets/sample/${c.dataset.key}`, { method: "POST" });
      await loadDatasets();
      await selectDataset(ds.id);
    } catch (err) { toast(err.message, "err"); c.style.opacity = "1"; }
  }));
  $$(".ds-row", v).forEach(r => r.addEventListener("click", () => selectDataset(r.dataset.id)));
}

async function uploadFile(file) {
  toast(`正在上传 ${file.name}…`);
  const fd = new FormData();
  fd.append("file", file);
  try {
    const ds = await api("/api/upload", { method: "POST", body: fd });
    await loadDatasets();
    await selectDataset(ds.id, { silent: true });
    toast(`已载入「${ds.name}」:${ds.rows} 行 × ${ds.cols} 列`, "ok");
  } catch (err) { toast(err.message, "err"); }
}

/* ---------------- 数据页 ---------------- */
async function viewData(id) {
  S.model = null;
  try { S.ds = await api(`/api/datasets/${id}?page=${S.dataPage}&page_size=50`); }
  catch { location.hash = "#/"; return; }
  renderTree($("#modelSearch").value);
  renderDataView();
}

function renderDataView() {
  const ds = S.ds;
  setTopbar("数据", ds.name, `
    <span class="chip-note">${ds.rows} 行 × ${ds.cols} 列</span>
    <button class="btn" id="btnProc">⚙ 数据处理</button>
    <button class="btn primary" id="btnAnalyze">开始分析 →</button>`);
  $("#btnAnalyze").addEventListener("click", () => { location.hash = "#/"; });
  $("#btnProc").addEventListener("click", openProcessModal);

  const p = ds.preview;
  const colMeta = Object.fromEntries(ds.columns.map(c => [c.name, c]));
  const head = p.columns.map(c => {
    const meta = colMeta[c];
    return `<th title="点击修改变量类型" data-col="${esc(c)}">${esc(c)}<br>${vtBadge(meta.dtype)}</th>`;
  }).join("");
  const body = p.rows.map((row, i) =>
    `<tr><td class="rowidx">${(p.page - 1) * p.pageSize + i + 1}</td>${row.map(cell => {
      const isNum = typeof cell === "number";
      return `<td class="${isNum ? "num" : ""}${cell === null ? " empty" : ""}">${cell === null ? "—" : esc(cell)}</td>`;
    }).join("")}</tr>`).join("");

  const chips = ds.columns.map(c => `
    <span class="col-chip" data-col="${esc(c.name)}" title="点击切换类型">
      <b>${esc(c.name)}</b>${vtBadge(c.dtype)}
    </span>`).join("");

  $("#view").innerHTML = `
  <div class="page-pad">
    <div class="toolbar">
      <span class="tb-title">数据浏览</span>
      <span class="tb-meta">${ds.source === "sample" ? "示例数据" : "上传数据"} · 缺失值显示为 —</span>
      <span class="spacer"></span>
    </div>
    <div class="card data-table-wrap">
      <table class="grid"><thead><tr><th class="rowidx">#</th>${head}</tr></thead><tbody>${body}</tbody></table>
      <div class="pager">
        <span>共 ${p.total} 行 · 第 ${p.page} / ${Math.max(1, Math.ceil(p.total / p.pageSize))} 页</span>
        <button id="pgPrev" ${p.page <= 1 ? "disabled" : ""}>上一页</button>
        <button id="pgNext" ${p.page * p.pageSize >= p.total ? "disabled" : ""}>下一页</button>
      </div>
    </div>
    <div class="card col-panel">
      <h4>变量类型 <span style="font-weight:400;font-size:12px;color:var(--ink-3)">点击变量可切换定量 / 定类 / 定级类型,分析时按类型校验</span></h4>
      <div class="col-chips">${chips}</div>
    </div>
  </div>`;

  $("#pgPrev").addEventListener("click", () => { S.dataPage--; viewData(ds.id); });
  $("#pgNext").addEventListener("click", () => { S.dataPage++; viewData(ds.id); });
  $$("th[data-col]").forEach(th => th.addEventListener("click", () => cycleType(th.dataset.col)));
  $$(".col-chip").forEach(ch => ch.addEventListener("click", () => cycleTypeMenu(ch)));
}

async function cycleType(col) {
  const order = ["quant", "cat", "ordinal"];
  const cur = S.ds.columns.find(c => c.name === col).dtype;
  const next = order[(order.indexOf(cur) + 1) % order.length] ?? "quant";
  await setType(col, next);
}

function cycleTypeMenu(chip) {
  const col = chip.dataset.col;
  const pop = openPopover(chip, `<div class="pop-head">${esc(col)} · 变量类型</div><div class="pop-list">${
    ["quant", "cat", "ordinal"].map(t => `<button class="pop-item" data-t="${t}">${vtBadge(t)} ${VT_LABEL[t]}</button>`).join("")}</div>`);
  $$(".pop-item", pop).forEach(b => b.addEventListener("click", () => { setType(col, b.dataset.t); closePopover(); }));
}

async function setType(col, t) {
  try {
    await api(`/api/datasets/${S.ds.id}/columns`, { method: "PUT", body: { column: col, dtype: t } });
    S.ds = await api(`/api/datasets/${S.ds.id}?page=${S.dataPage}`);
    renderDataView();
    toast(`「${col}」→ ${VT_LABEL[t]}`, "ok");
  } catch (err) { toast(err.message, "err"); }
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
    fill_missing: `<div class="opt-row"><label class="field-label">目标列(可多选)</label><select id="pfCols" class="select" multiple size="6">${colOpts}</select></div>
      <div class="opt-row"><label class="field-label">填充方式</label><select id="pfMethod" class="select">
        <option value="mean">均值(定量)</option><option value="median">中位数(定量)</option><option value="mode">众数</option>
        <option value="ffill">前一有效值</option><option value="constant">常数</option></select></div>
      <div class="opt-row"><label class="field-label">常数值(方式选常数时)</label><input id="pfValue" class="input" value="0"></div>`,
    drop_missing: `<div class="opt-row"><label class="field-label">检查列(留空=所有列)</label><select id="pfCols" class="select" multiple size="6">${colOpts}</select></div>
      <div class="opt-row"><label class="field-label">删除条件</label><select id="pfHow" class="select"><option value="any">任一列缺失</option><option value="all">全部列缺失</option></select></div>`,
    winsorize: `<div class="opt-row"><label class="field-label">定量列(可多选)</label><select id="pfCols" class="select" multiple size="6">${quantOpts}</select></div>
      <div class="opt-row"><label class="field-label">判定方法</label><select id="pfMethod" class="select"><option value="3sigma">3σ 法则</option><option value="iqr">四分位 IQR</option></select></div>`,
    standardize: `<div class="opt-row"><label class="field-label">定量列(可多选)</label><select id="pfCols" class="select" multiple size="6">${quantOpts}</select></div>
      <div class="opt-row"><label class="field-label">方法</label><select id="pfMethod" class="select">
        <option value="zscore">Z 标准化(均值0方差1)</option><option value="minmax">归一化 [0,1]</option>
        <option value="center">中心化</option><option value="robust">稳健标准化(中位数/IQR)</option></select></div>`,
    dummy: `<div class="opt-row"><label class="field-label">定类列</label><select id="pfCol" class="select">${colOpts}</select></div>`,
    recode_bins: `<div class="opt-row"><label class="field-label">定量列</label><select id="pfCol" class="select">${quantOpts}</select></div>
      <div class="opt-row"><label class="field-label">组数</label><input id="pfBins" class="input" type="number" value="3" min="2" max="10"></div>
      <div class="opt-row"><label class="field-label">新列名</label><input id="pfNew" class="input" value="分组"></div>`,
    compute: `<div class="opt-row"><label class="field-label">公式(可用列名与 + - * / ( ),例: 月薪 * 1.2 + 工龄 * 100)</label>
      <textarea id="pfExpr" class="textarea" placeholder="月薪 / 1000"></textarea></div>
      <div class="opt-row"><label class="field-label">新列名</label><input id="pfNew" class="input" value="新变量"></div>`,
    filter: `<div class="opt-row"><label class="field-label">筛选表达式(例: 年龄 > 30 and 性别 == '男')</label>
      <textarea id="pfExpr" class="textarea" placeholder="总体满意度 >= 4"></textarea></div>`,
    sort: `<div class="opt-row"><label class="field-label">排序列(可多选,按顺序)</label><select id="pfCols" class="select" multiple size="5">${colOpts}</select></div>
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
  if (!S.ds) {
    const ds = await ensureDs();
    if (!ds) { location.hash = "#/"; toast("请先载入数据集"); return; }
  }
  renderTree($("#modelSearch").value);
  setTopbar(found.category, found.name, `
    ${S.ds ? `<span class="chip-note">数据集:${esc(S.ds.name)}</span>` : ""}
    <button class="btn" id="btnData">查看数据</button>`);
  $("#btnData").addEventListener("click", () => location.hash = `#/data/${S.ds.id}`);

  // 初始化槽位与选项(切换模型时重置,同模型内保留已选)
  if (!prevModel || prevModel.id !== found.id) {
    S.slots = {};
    for (const sl of found.slots) S.slots[sl.key] = sl.multiple ? [] : "";
    S.options = {};
    for (const o of found.options) S.options[o.key] = o.default;
  }

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
    <div class="results" id="results">${resultEmptyHtml(found)}</div>
  </div>`;

  // 槽位事件
  $$(".slot-box").forEach(box => {
    const key = box.dataset.slot;
    box.addEventListener("click", (e) => {
      if (e.target.closest(".x")) return;
      openVarPicker(box, key);
    });
  });
  // 移除变量(事件委托)
  $("#view").addEventListener("click", (e) => {
    const x = e.target.closest(".x[data-slot]");
    if (!x) return;
    e.stopPropagation();
    const { slot, col } = x.dataset;
    if (S.model.slots.find(s => s.key === slot)?.multiple) S.slots[slot] = S.slots[slot].filter(c => c !== col);
    else S.slots[slot] = "";
    viewModelRefreshPanel();
  }, { once: true });

  // 选项事件
  $$("[data-opt]").forEach(el => el.addEventListener("change", () => {
    const k = el.dataset.opt;
    S.options[k] = el.type === "checkbox" ? el.checked : el.value;
  }));

  $("#btnRun").addEventListener("click", runModel);
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
    const lv = c.levels ? ` · ${c.levels.length} 水平` : "";
    return `<button class="pop-item ${inAnother ? "" : ""}" data-col="${esc(c.name)}">
      ${vtBadge(c.dtype)}<span>${esc(c.name)}</span><span class="pi-meta">${lv}</span></button>`;
  }).join("") || `<div style="padding:14px;font-size:12.5px;color:var(--ink-3)">当前数据集没有符合该槽位类型的变量,可到数据页修改变量类型。</div>`;
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
    <h4>在左侧配置变量,点「运行分析」</h4>
    <p>结果将在此展示:智能解读 → 结果表格 → 可视化图表。<br>当前模型:${esc(model.name)}</p>
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

/* ---------------- 结果渲染 ---------------- */
let chartObservers = [];
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
      <p>运行任意模型后,本次会话的分析结果会自动汇总到这里,并可一键导出为可打印的 HTML 报告。</p>
      <a class="btn primary" href="#/">去运行第一个分析 →</a></div>`;
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
        <button class="btn primary" id="btnReportExport">⬇ 导出 HTML 报告</button>
        <button class="btn ghost" id="btnReportClear">清空记录</button>
      </div>
    </div>
    <nav class="report-toc card">
      <b>目录</b>
      <ol>${S.history.map((e, i) => `<li><a href="#rp${i}">${esc(e.modelName)}<span class="toc-meta">${esc(e.dsName)}</span></a></li>`).join("")}</ol>
    </nav>
    ${html}`;
  $("#btnReportExport").addEventListener("click", exportReportHtml);
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
    </section>`;
  }).join("");
  const doc = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>OpenSPSS 分析报告</title><style>
body{font-family:"Microsoft YaHei",system-ui,sans-serif;color:#1e293b;max-width:960px;margin:0 auto;padding:32px 24px;line-height:1.65}
.rp-cover{text-align:center;padding:48px 0 24px;border-bottom:3px solid #4653E8;margin-bottom:24px}
.rp-cover h1{margin:0 0 8px;font-size:26px;letter-spacing:2px}
.rp-meta{color:#64748b;font-size:12.5px;margin:2px 0}
.rp-slots{background:#f1f5f9;border-radius:8px;padding:8px 12px;font-size:12.5px;color:#475569;margin:8px 0}
.rp-toc{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 24px;margin-bottom:28px;font-size:13.5px}
.rp-toc a{color:#4653E8;text-decoration:none}
.rp-item{margin-bottom:36px}
.rp-item h2{border-left:4px solid #4653E8;padding-left:10px;font-size:19px}
.rp-item h4{margin:18px 0 6px;font-size:14.5px;color:#334155}
.rp-interp{background:#eef2ff;border-radius:10px;padding:10px 14px;margin:10px 0}
.rp-table{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0}
.rp-table th{background:#f1f5f9;text-align:left}
.rp-table th,.rp-table td{border:1px solid #e2e8f0;padding:5px 9px}
.rp-table tr:nth-child(even) td{background:#fafbfe}
.rp-chart{max-width:100%;border:1px solid #eef2f7;border-radius:8px;margin:4px 0 12px}
.rp-note{color:#64748b;font-size:12px}
footer{border-top:1px solid #e2e8f0;margin-top:40px;padding-top:12px;color:#94a3b8;font-size:12px;text-align:center}
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
          if (!confirm("停止本地 OpenSPSS 服务并退出?")) return;
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
  window.addEventListener("hashchange", route);
  route();
}
boot();
