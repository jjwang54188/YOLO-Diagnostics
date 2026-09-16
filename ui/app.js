// 页面交互只负责收集配置、调用接口和显示状态，诊断计算在 Python 中。
const $ = (id) => document.getElementById(id);
const token = document.querySelector('meta[name="app-token"]').content;
const statusNames = {
  running: "运行中",
  cancelling: "正在取消",
  cancelled: "已取消",
  completed: "已完成",
  failed: "失败",
  interrupted: "已中断",
};
const pageText = {
  new: [
    "新建一次模型分析",
    "选择模型和带标注的数据，找到哪些目标识别差，以及可以复核的证据。",
  ],
  task: ["任务进度", "查看真实处理进度与运行日志。完成后，直接打开报告。"],
  history: ["历史分析报告", "配置、日志和报告都会保留，方便复查和重复实验。"],
  materials: [
    "数据清单与诊断依据",
    "准备可复核的材料，了解结论怎样得出、还缺哪些证据。",
  ],
  ai: [
    "AI 辅助诊断",
    "连接你自己的 AI，先预览证据，再进一步理解与制定修复实验。",
  ],
  help: ["使用与开发指南", "先了解怎么使用，再了解每个文件负责什么。"],
};
let state = { jobs: [], active: null },
  currentJob = null,
  selectedJob = localStorage.getItem("diagnostics-job"),
  page = "new",
  fileTarget = null,
  directory = "",
  home = "",
  loaded = false;
const esc = (s) =>
  String(s).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      "X-Diagnostics-Token": token,
      "Content-Type": "application/json",
      ...options.headers,
    },
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "操作失败，请重试。");
  return data;
}
function notify(message) {
  $("notification").textContent = message;
  $("notification").hidden = !message;
}
function showPage(name) {
  page = name;
  document
    .querySelectorAll(".page")
    .forEach((e) => (e.hidden = e.id !== `page-${name}`));
  document
    .querySelectorAll(".nav")
    .forEach((e) => e.classList.toggle("active", e.dataset.page === name));
  $("page-title").textContent = pageText[name][0];
  $("page-description").textContent = pageText[name][1];
  if (name === "task" && selectedJob) loadJob(selectedJob);
  if (name === "history") renderHistory();
  if (name === "materials") loadGuide();
  if (name === "ai") loadAI();
}
document
  .querySelectorAll("[data-page]")
  .forEach((button) =>
    button.addEventListener("click", () => showPage(button.dataset.page)),
  );
function fillForm(config) {
  for (const key of ["context", "metadata", "train_run", "baseline_run"])
    $(key).value = "";
  for (const [key, value] of Object.entries(config)) {
    if ($(key) && $(key).form === $("analysis-form"))
      $(key).value = value ?? "";
  }
  previewDataset();
}
function configFromForm() {
  return Object.fromEntries(new FormData($("analysis-form")));
}

async function previewDataset() {
  const path = $("data").value.trim();
  if (!path) {
    $("dataset-preview").textContent = "选择配置后，这里会显示类别和数据分组。";
    return;
  }
  try {
    const data = await api("/api/dataset?path=" + encodeURIComponent(path));
    if (path !== $("data").value.trim()) return;
    const names = Array.isArray(data.names)
      ? data.names
      : Object.values(data.names);
    $("dataset-preview").innerHTML =
      `<strong>已读取 ${names.length} 个类别</strong><div>分组：${["train", "val", "test"].filter((k) => data[k]).join(" / ") || "未提供"}</div><div class="chips">${names.map((n) => `<span class="chip">${esc(n)}</span>`).join("")}</div>`;
  } catch (error) {
    $("dataset-preview").textContent = error.message;
  }
}
$("data").addEventListener("change", previewDataset);
$("analysis-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  notify("");
  $("start").disabled = true;
  try {
    const result = await api("/api/jobs", {
      method: "POST",
      body: JSON.stringify(configFromForm()),
    });
    selectedJob = result.id;
    localStorage.setItem("diagnostics-job", selectedJob);
    showPage("task");
    await refresh();
  } catch (error) {
    notify(error.message);
    $("start").disabled = Boolean(state.active);
  }
});
$("demo").onclick = () => {
  if (state.demo) {
    fillForm(state.demo);
    notify("");
  } else notify("这台电脑未配置演示文件。请手动选择你的模型和数据集。");
};

async function loadJob(identifier) {
  try {
    const job = await api("/api/jobs/" + encodeURIComponent(identifier));
    if (identifier !== selectedJob) return;
    currentJob = job;
    $("no-task").hidden = true;
    $("task-content").hidden = false;
    $("task-name").textContent = job.name;
    $("task-status").textContent = statusNames[job.status] || job.status;
    const p = job.progress || {};
    $("task-phase").textContent =
      job.status === "running"
        ? p.total
          ? `${p.phase}：${p.done} / ${p.total} 张图片`
          : p.phase || "正在准备"
        : job.status === "completed"
          ? "分析已完成，报告可以随时打开。"
          : job.status === "cancelling"
            ? "正在停止分析进程…"
            : "任务已停止，已有日志保留。";
    if (job.status === "running" && p.total) {
      $("task-progress").value = p.done;
      $("task-progress").max = p.total;
    } else if (job.status === "running" || job.status === "cancelling") {
      $("task-progress").removeAttribute("value");
    } else {
      $("task-progress").max = 1;
      $("task-progress").value = job.status === "completed" ? 1 : 0;
    }
    const c = job.config;
    $("task-config").innerHTML = [
      ["问题模型", c.model],
      ["对照模型", c.baseline || "未使用"],
      ["数据配置", c.data],
      [
        "分析条件",
        `${c.split} / imgsz=${c.imgsz} / conf=${c.conf} / IoU=${c.match_iou} / ${c.device === "cpu" ? "CPU" : "GPU 0"}`,
      ],
    ]
      .map(([k, v]) => `<div><b>${k}</b>${esc(v)}</div>`)
      .join("");
    $("cancel").hidden = job.status !== "running";
    $("task-ai").hidden = job.status !== "completed";
    $("open-report").hidden = !job.report;
    if (job.report) $("open-report").href = job.report;
    let error = job.error || "";
    if (job.status === "failed" && job.log?.includes("CUDA out of memory"))
      error = "显存不足。可换 CPU 或降低推理尺寸后另跑一次；详细信息见日志。";
    $("task-error").hidden = !error;
    $("task-error").textContent = error;
    const log = $("task-log"),
      bottom = log.scrollHeight - log.scrollTop - log.clientHeight < 50;
    log.textContent =
      job.log ||
      (job.imported
        ? "此任务从已有报告导入，没有原始运行日志。"
        : "等待程序输出…");
    if (bottom) log.scrollTop = log.scrollHeight;
  } catch (error) {
    notify(error.message);
  }
}
$("cancel").onclick = async () => {
  if (
    !currentJob ||
    !confirm("取消会停止当前分析，保留已生成的文件和日志。确定取消吗？")
  )
    return;
  try {
    await api(`/api/jobs/${currentJob.id}/cancel`, { method: "POST" });
    await loadJob(currentJob.id);
  } catch (error) {
    notify(error.message);
  }
};
$("reuse").onclick = () => {
  if (currentJob) {
    fillForm(currentJob.config);
    showPage("new");
  }
};

function renderHistory() {
  const search = $("history-search").value.toLowerCase(),
    filter = $("history-filter").value;
  const jobs = state.jobs.filter(
    (job) =>
      (job.name + " " + job.config.model + " " + job.config.baseline)
        .toLowerCase()
        .includes(search) &&
      (filter === "all" ||
        job.status === filter ||
        (filter === "failed" &&
          ["failed", "cancelled", "interrupted"].includes(job.status))),
  );
  $("history-list").innerHTML = jobs.length
    ? jobs
        .map(
          (job) =>
            `<article class="history-item"><div><span class="badge">${statusNames[job.status] || esc(job.status)}</span><h3>${esc(job.name)}</h3><p>${esc(new Date(job.created).toLocaleString())} · ${esc(job.config.split)} · ${job.config.imgsz}px<br>${esc(job.config.model.split("/").pop())}${job.config.baseline ? " 对比 " + esc(job.config.baseline.split("/").pop()) : ""}</p></div><div class="history-actions">${job.report ? `<a class="primary button" href="${esc(job.report)}" target="_blank" rel="noopener">打开报告 ↗</a>` : ""}<button data-job="${esc(job.id)}">任务详情</button><button data-reuse="${esc(job.id)}">复用设置</button></div></article>`,
        )
        .join("")
    : '<div class="empty"><h2>还没有匹配的任务</h2><p>完成一次分析后，报告会出现在这里。</p></div>';
  $("history-list")
    .querySelectorAll("[data-job]")
    .forEach(
      (button) =>
        (button.onclick = () => {
          selectedJob = button.dataset.job;
          localStorage.setItem("diagnostics-job", selectedJob);
          showPage("task");
        }),
    );
  $("history-list")
    .querySelectorAll("[data-reuse]")
    .forEach(
      (button) =>
        (button.onclick = () => {
          fillForm(
            state.jobs.find((j) => j.id === button.dataset.reuse).config,
          );
          showPage("new");
        }),
    );
}
$("history-search").oninput = renderHistory;
$("history-filter").onchange = renderHistory;

async function browse(path) {
  $("browse-error").textContent = "";
  $("file-list").textContent = "正在读取文件夹…";
  try {
    const result = await api(
      `/api/browse?kind=${["train_run", "baseline_run"].includes(fileTarget) ? "directory" : fileTarget === "baseline" ? "model" : fileTarget}&path=${encodeURIComponent(path)}`,
    );
    directory = result.path;
    $("browse-path").value = directory;
    $("browse-parent").dataset.path = result.parent;
    $("file-list").replaceChildren();
    if (!result.entries.length)
      $("file-list").textContent = "这里没有可选择的文件，请进入其他文件夹。";
    for (const entry of result.entries) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = entry.directory ? "folder" : "file";
      button.textContent = (entry.directory ? "▸  " : "◇  ") + entry.name;
      button.onclick = () => {
        if (entry.directory) browse(entry.path);
        else {
          $(fileTarget).value = entry.path;
          $("file-dialog").close();
          if (fileTarget === "data") previewDataset();
        }
      };
      $("file-list").append(button);
    }
  } catch (error) {
    $("file-list").textContent = "";
    $("browse-error").textContent = error.message;
  }
}
document.querySelectorAll("[data-browse]").forEach(
  (button) =>
    (button.onclick = () => {
      fileTarget = button.dataset.browse;
      $("file-title").textContent = {
        model: "选择问题模型（.pt）",
        baseline: "选择对照模型（.pt）",
        data: "选择数据集配置（.yaml / .yml）",
        context: "选择补充资料 JSON",
        metadata: "选择场景 CSV",
        train_run: "选择当前模型训练目录",
        baseline_run: "选择对照模型训练目录",
      }[fileTarget];
      $("choose-directory").hidden = !["train_run", "baseline_run"].includes(
        fileTarget,
      );
      $("file-dialog").showModal();
      const current = $(fileTarget).value;
      const parent = current.includes("/")
        ? current.slice(0, current.lastIndexOf("/"))
        : home;
      browse(parent || home);
    }),
);
$("choose-directory").onclick = () => {
  $(fileTarget).value = directory;
  $("file-dialog").close();
};
$("close-browser").onclick = () => $("file-dialog").close();
$("browse-form").onsubmit = (event) => {
  event.preventDefault();
  browse($("browse-path").value);
};
$("browse-parent").onclick = () => browse($("browse-parent").dataset.path);
document
  .querySelectorAll("[data-place]")
  .forEach(
    (button) =>
      (button.onclick = () =>
        browse(
          button.dataset.place === "home"
            ? home
            : home + "/" + button.dataset.place,
        )),
  );

async function refresh() {
  try {
    state = await api("/api/state");
    home = state.home;
    $("connection").textContent = "● 本地服务在线";
    if (!loaded) {
      fillForm(state.settings);
      if (!selectedJob || !state.jobs.some((j) => j.id === selectedJob))
        selectedJob = state.active;
      loaded = true;
    }
    $("demo").disabled = !state.demo;
    $("start").disabled = Boolean(state.active);
    $("ready-text").textContent = state.active
      ? "已有任务运行中，请等待或取消"
      : "配置完成后即可开始";
    if (page === "history") renderHistory();
    if (page === "task" && selectedJob) await loadJob(selectedJob);
  } catch (error) {
    $("connection").textContent = "○ 服务未连接";
    notify("无法连接本地服务。请重新运行启动程序，再刷新页面。");
  }
}
async function poll() {
  await refresh();
  setTimeout(poll, state.active ? 1500 : 5000);
}
poll();

async function loadGuide() {
  try {
    const guide = await api("/api/guide");
    $("data-guide").innerHTML = guide.data
      .map(
        (x) =>
          `<details open><summary>${esc(x.priority)} · ${esc(x.name)}</summary><p><b>格式：</b>${esc(x.format)}</p><p><b>能帮助判断：</b>${esc(x.use)}</p><p><b>当前支持：</b>${esc(x.support)}</p></details>`,
      )
      .join("");
    $("algorithm-guide").innerHTML = guide.methods
      .map(
        (x) =>
          `<details><summary>${esc(x.name)}</summary><p>${esc(x.algorithm)}</p><p><b>限制：</b>${esc(x.limit)}</p></details>`,
      )
      .join("");
  } catch (error) {
    notify(error.message);
  }
}
