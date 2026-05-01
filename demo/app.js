/* MetaJudge AI Dashboard — Application Logic */

const API_BASE = 'http://localhost:8000';

// DOM refs
const inputEl     = document.getElementById('input-text');
const modeEl      = document.getElementById('mode-select');
const runBtn      = document.getElementById('run-btn');
const spinner     = document.getElementById('spinner');
const progressW   = document.getElementById('progress-wrap');
const progressF   = document.getElementById('progress-fill');
const progressL   = document.getElementById('progress-label');
const logConsole  = document.getElementById('log-console');
const resultsEl   = document.getElementById('results-section');
const healthDot   = document.getElementById('health-dot');
const healthText  = document.getElementById('health-text');

// ── Health check ─────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    healthDot.classList.add('ok');
    healthDot.classList.remove('err');
    healthText.textContent = data.nvidia_key_set ? 'API Connected' : 'Key Missing';
  } catch {
    healthDot.classList.add('err');
    healthDot.classList.remove('ok');
    healthText.textContent = 'Backend Offline';
  }
}

// ── Logging ──────────────────────────────────────────────────────────────
function appendLog(kind, message) {
  if (!message) return;
  const line = document.createElement('div');
  line.className = `log-${kind || 'info'}`;
  line.textContent = message;
  logConsole.appendChild(line);
  logConsole.scrollTop = logConsole.scrollHeight;
}

function clearUI() {
  logConsole.innerHTML = '';
  logConsole.classList.add('active');
  progressW.classList.add('active');
  progressF.style.width = '0%';
  progressL.textContent = 'Initializing...';
  resultsEl.classList.remove('active');
  resultsEl.innerHTML = '';
}

// ── Progress ─────────────────────────────────────────────────────────────
function setProgress(value, label) {
  progressF.style.width = `${Math.round(value * 100)}%`;
  if (label) progressL.textContent = label;
}

// ── Run pipeline via SSE ─────────────────────────────────────────────────
async function runAnalysis() {
  const text = inputEl.value.trim();
  if (!text) { alert('Enter text to analyze.'); return; }

  runBtn.disabled = true;
  spinner.classList.add('active');
  clearUI();

  const mode = modeEl.value;
  const url = `${API_BASE}/analyze/stream?text=${encodeURIComponent(text)}&mode=${mode}`;

  try {
    const res = await fetch(url);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Request failed');
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const event = JSON.parse(line.slice(6));
          handleSSEEvent(event);
        } catch { /* skip malformed */ }
      }
    }
  } catch (err) {
    appendLog('err', `Error: ${err.message}`);
    // Fallback to sync endpoint
    appendLog('info', 'Falling back to synchronous endpoint...');
    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, mode }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const data = await res.json();
      setProgress(1, 'Complete');
      renderResults(data);
    } catch (e2) {
      appendLog('err', `Sync fallback failed: ${e2.message}`);
    }
  } finally {
    runBtn.disabled = false;
    spinner.classList.remove('active');
  }
}

function handleSSEEvent(event) {
  if (event.stage === 'done') {
    setProgress(1, 'Pipeline complete');
    if (event.final_output) renderResults(event.final_output);
    return;
  }

  // Stage progress events
  if (event.status === 'complete') {
    appendLog('done', `✓ ${event.stage} (${event.duration_ms?.toFixed(0) || '?'}ms)`);
    // Estimate progress from stages
    const stages = ['atomicizer','consistency_checker','query_generator','retriever','judge','cove_loop','editor'];
    const idx = stages.indexOf(event.stage);
    if (idx >= 0) setProgress((idx + 1) / stages.length, `${event.stage} complete`);
  }

  // Log events from on_event callback
  if (event.type === 'progress') {
    setProgress(event.value, event.label);
  } else if (event.type === 'log') {
    appendLog(event.kind, event.message);
  }
}

// ── Render results ───────────────────────────────────────────────────────
function renderResults(data) {
  resultsEl.classList.add('active');
  const results = data.results || [];
  const corrections = data.corrections || [];

  const supported = results.filter(r => r.verdict === 'SUPPORTED').length;
  const contradicted = results.filter(r => ['CONTRADICTED','INTERNAL_CONTRADICTION'].includes(r.verdict)).length;
  const insufficient = results.filter(r => r.verdict === 'INSUFFICIENT_EVIDENCE').length;

  let html = `
    <div class="card">
      <h2><span class="icon">📊</span> Results</h2>
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-value">${results.length}</div>
          <div class="stat-label">Facts Checked</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" style="color:var(--accent-green)">${supported}</div>
          <div class="stat-label">Supported</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" style="color:var(--accent-red)">${contradicted}</div>
          <div class="stat-label">Contradicted</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" style="color:var(--accent-amber)">${insufficient}</div>
          <div class="stat-label">Insufficient</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" style="color:var(--accent-cyan)">${corrections.length}</div>
          <div class="stat-label">Corrections</div>
        </div>
      </div>`;

  // Per-fact breakdown
  for (const r of results) {
    const v = r.verdict || 'INSUFFICIENT_EVIDENCE';
    const vClass = v === 'SUPPORTED' ? 'supported'
      : v === 'CONTRADICTED' ? 'contradicted'
      : v === 'INTERNAL_CONTRADICTION' ? 'internal' : 'insufficient';
    const vLabel = v === 'SUPPORTED' ? 'v-supported'
      : v === 'CONTRADICTED' ? 'v-contradicted'
      : v === 'INTERNAL_CONTRADICTION' ? 'v-internal' : 'v-insufficient';

    html += `
      <div class="fact-item ${vClass}">
        <div class="fact-text">${escapeHtml(r.fact || '')}</div>
        <span class="fact-verdict ${vLabel}">${v.replace('_', ' ')}</span>
        ${r.cove_applied ? `<span class="fact-verdict v-internal">CoVe: ${r.cove_meta_verdict || '?'}</span>` : ''}
        ${r.reasoning ? `<div class="fact-meta">${escapeHtml(r.reasoning.substring(0, 150))}</div>` : ''}
      </div>`;
  }

  // Corrections
  if (corrections.length > 0) {
    html += `<div class="correction-box"><h3>✏️ Corrections Applied</h3>`;
    for (const c of corrections) {
      html += `
        <div class="diff-old">− ${escapeHtml(c.error_span || '')}</div>
        <div class="diff-new">+ ${escapeHtml(c.correction || '')}</div>
        <div class="fact-meta">Source: ${escapeHtml(c.source_url || 'N/A')}</div><br>`;
    }
    html += `</div>`;
  }

  // Corrected summary
  if (data.corrected && data.corrected !== data.original) {
    html += `
      <div class="correction-box" style="margin-top:16px">
        <h3>📝 Corrected Summary</h3>
        <div class="fact-text">${escapeHtml(data.corrected)}</div>
      </div>`;
  }

  html += `</div>`;
  resultsEl.innerHTML = html;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ── Examples ─────────────────────────────────────────────────────────────
function loadExample(idx) {
  const examples = [
    "BERT, introduced by Google in 2018, uses a bidirectional transformer encoder. It was pre-trained on BookCorpus and English Wikipedia, and achieved 80.5% F1 on the SQuAD 2.0 benchmark. The paper was authored by Devlin et al. and published at NAACL 2019.",
    "GPT-4, released by OpenAI in March 2022, is a large multimodal model that accepts image and text inputs. It achieved 86.4% on the MMLU benchmark, surpassing all previous models.",
    "The LoRA paper by Hu et al. from Microsoft proposed a parameter-efficient fine-tuning method using low-rank decomposition with a rank of 8. It demonstrated 91.3% accuracy on MNLI when applied to GPT-3."
  ];
  if (examples[idx]) inputEl.value = examples[idx];
}

// ── Init ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  setInterval(checkHealth, 30000);
  runBtn.addEventListener('click', runAnalysis);
});
