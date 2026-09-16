// AI 接口是单独的辅助意见通道。任何表单变化都使旧预览失效，防止误发别的内容。
let aiRevision = 0;
let aiPreview = null,
  aiAnswer = null,
  aiRunning = false;
function invalidateAI() {
  aiRevision++;
  aiPreview = null;
  $("ai-preview-panel").hidden = true;
  $("ai-consent").checked = false;
  $("ai-send").disabled = true;
}
async function loadAI() {
  if (aiRunning) return;
  try {
    const settings = await api("/api/ai/settings");
    $("ai-endpoint").value = settings.endpoint;
    $("ai-model").value = settings.model;
    $("ai-max-tokens").value = settings.max_tokens;
    $("ai-config-status").textContent = settings.has_key
      ? "Key 已保留在当前服务内存，未保存到文件。"
      : "当前未设置 Key；无鉴权的本机接口可以留空。";
    const selected = $("ai-job").value || selectedJob;
    const jobs = state.jobs.filter((j) => j.status === "completed");
    $("ai-job").replaceChildren(...jobs.map((j) => new Option(j.name, j.id)));
    if (jobs.some((j) => j.id === selected)) $("ai-job").value = selected;
    await loadAIEvidence();
  } catch (error) {
    $("ai-status").textContent = error.message;
  }
}
async function loadAIEvidence() {
  invalidateAI();
  const id = $("ai-job").value;
  $("ai-images").replaceChildren();
  $("ai-answer").textContent = "";
  $("ai-export").hidden = true;
  if (!id) {
    $("ai-status").textContent = "请先完成一次模型分析。";
    return;
  }
  try {
    const [images, history] = await Promise.all([
      api(`/api/ai/images?job_id=${encodeURIComponent(id)}`),
      api(`/api/ai/history?job_id=${encodeURIComponent(id)}`),
    ]);
    if (id !== $("ai-job").value) return;
    $("ai-images").innerHTML = images
      .map(
        (i) =>
          `<label class="image-option"><input type="checkbox" value="${i.id}"/><img src="${esc(i.url)}" loading="lazy" alt="图片 ${i.id} 预览"/><span>#${i.id} ${esc(i.name)}</span></label>`,
      )
      .join("");
    $("ai-images")
      .querySelectorAll("input")
      .forEach(
        (input) =>
          (input.onchange = () => {
            if ($("ai-images").querySelectorAll("input:checked").length > 4) {
              input.checked = false;
              $("ai-status").textContent = "最多选择 4 张图片。";
            }
            invalidateAI();
          }),
      );
    $("ai-previous").replaceChildren(
      new Option("不附加历史回答", ""),
      ...history.map(
        (h) => new Option(`${h.created} · ${h.question.slice(0, 45)}`, h.id),
      ),
    );
    $("ai-history").innerHTML =
      history
        .map(
          (h) =>
            `<details><summary>${esc(h.created)} · ${esc(h.model)} · ${esc(h.question.slice(0, 70))}</summary><p>AI 辅助意见，未经人工验证。</p><pre>${esc(h.answer)}</pre><small>终止原因：${esc(h.finish_reason || "未提供")}</small></details>`,
        )
        .join("") || "本报告暂无 AI 意见。";
    $("ai-status").textContent = "尚未发送；可以先检查报告并选择证据。";
  } catch (error) {
    $("ai-status").textContent = error.message;
  }
}
$("ai-config-form").onsubmit = async (event) => {
  event.preventDefault();
  invalidateAI();
  try {
    const settings = await api("/api/ai/settings", {
      method: "POST",
      body: JSON.stringify({
        endpoint: $("ai-endpoint").value,
        model: $("ai-model").value,
        api_key: $("ai-key").value,
        clear_key: $("ai-clear-key").checked,
        max_tokens: $("ai-max-tokens").value,
      }),
    });
    $("ai-key").value = "";
    $("ai-clear-key").checked = false;
    $("ai-config-status").textContent =
      `连接配置已保存，尚未连接服务商。${settings.has_key ? "Key 仅保留在服务内存。" : "没有配置 Key。"}`;
  } catch (error) {
    $("ai-config-status").textContent = error.message;
  }
};
$("ai-job").onchange = loadAIEvidence;
for (const id of [
  "ai-question",
  "ai-include-context",
  "ai-previous",
  "ai-endpoint",
  "ai-model",
  "ai-key",
  "ai-max-tokens",
  "ai-clear-key",
])
  $(id).addEventListener("input", invalidateAI);
$("ai-consent").onchange = () =>
  ($("ai-send").disabled = !aiPreview || !$("ai-consent").checked || aiRunning);
$("ai-preview").onclick = async () => {
  invalidateAI();
  $("ai-preview").disabled = true;
  const revision = aiRevision;
  const job = $("ai-job").value;
  try {
    const result = await api("/api/ai/preview", {
      method: "POST",
      body: JSON.stringify({
        job_id: job,
        question: $("ai-question").value,
        include_context: $("ai-include-context").checked,
        previous_id: $("ai-previous").value,
        image_ids: [...$("ai-images").querySelectorAll("input:checked")].map(
          (i) => +i.value,
        ),
      }),
    });
    if (job !== $("ai-job").value || revision !== aiRevision) return;
    aiPreview = result;
    $("ai-destination").textContent =
      `接收地址：${result.endpoint}；模型：${result.model}；证据与提问 ${result.text.length} 字符，图片 ${result.attachments.length} 张。`;
    $("ai-system").textContent = result.system_prompt;
    $("ai-packet").textContent = result.text;
    $("ai-attachments").innerHTML = result.attachments
      .map(
        (i) =>
          `<div class="image-option"><img src="${i.url}" alt="将发送图片 ${i.id}"/><span>将发送图片 #${i.id}<br/>SHA-256 ${esc(i.sha256)}</span></div>`,
      )
      .join("");
    $("ai-previous-preview").hidden = !result.previous_answer;
    $("ai-previous-text").textContent = result.previous_answer
      ? `此前提问：${result.previous_question}\n\n此前回答：${result.previous_answer}`
      : "";
    $("ai-preview-panel").hidden = false;
    $("ai-status").textContent = "预览已生成，尚未发送。请逐项核对。";
  } catch (error) {
    $("ai-status").textContent = error.message;
  } finally {
    $("ai-preview").disabled = false;
  }
};
$("ai-send").onclick = async () => {
  if (!aiPreview || !$("ai-consent").checked || aiRunning) return;
  const id = aiPreview.id;
  aiRunning = true;
  $("ai-send").disabled = true;
  $("ai-preview").disabled = true;
  $("ai-status").textContent =
    "AI 正在分析，请等待；不会自动重试。可切换页面，服务仍会保存成功返回的意见。";
  try {
    const result = await api("/api/ai/send", {
      method: "POST",
      body: JSON.stringify({ preview_id: id, consent: true }),
    });
    aiAnswer = result;
    $("ai-answer").textContent =
      `AI 辅助意见，未经人工验证\n${result.model} · ${result.created}\n\n${result.answer}`;
    $("ai-export").hidden = false;
    $("ai-status").textContent =
      result.finish_reason === "length"
        ? "回答达到长度上限，可能不完整。已保存，可调高输出上限后继续追问。"
        : "AI 意见已保存到本机。请根据原始证据复核后再实施建议。";
  } catch (error) {
    $("ai-status").textContent = error.message;
  } finally {
    aiRunning = false;
    invalidateAI();
    $("ai-preview").disabled = false;
  }
};
$("ai-export").onclick = () => {
  if (!aiAnswer) return;
  const content = `AI 辅助意见，未经人工验证\n模型：${aiAnswer.model}\n时间：${aiAnswer.created}\n问题：${aiAnswer.question}\n\n${aiAnswer.answer}`;
  const url = URL.createObjectURL(
    new Blob([content], { type: "text/plain;charset=utf-8" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = `ai-${aiAnswer.id}.txt`;
  a.click();
  URL.revokeObjectURL(url);
};

$("ai-deepseek").onclick = () => {
  invalidateAI();
  $("ai-endpoint").value = "https://api.deepseek.com/chat/completions";
  $("ai-model").value = "deepseek-flash";
  $("ai-max-tokens").value = "8192";
  $("ai-key").value = "";
  $("ai-clear-key").checked = true;
  $("ai-config-status").textContent =
    "已填入 DeepSeek 4.1 官方配置。请填写 Key 并保存；尚未发送。";
};
