/* OpenSPSS 前端基础:全局状态、工具函数、弹窗/气泡外壳(由 app.js 拆分) */
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

