import {
  ArrowClockwise,
  CheckCircle,
  ClipboardText,
  ClockCounterClockwise,
  FileText,
  Lightning,
  Pulse,
  Play,
  ShieldCheck,
  WarningCircle,
  XCircle,
} from "@phosphor-icons/react";
import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

const EXAMPLES = [
  {
    label: "BERT",
    text: "BERT, introduced by Google in 2018, uses a bidirectional transformer encoder. It was pre-trained on BookCorpus and English Wikipedia, and achieved 80.5% F1 on the SQuAD 2.0 benchmark. The paper was authored by Devlin et al.",
  },
  {
    label: "FActScore",
    text: "FActScore was proposed by Lee et al. in 2022 as a framework for evaluating factual precision. It decomposes model outputs into atomic facts and evaluates each independently against a knowledge source.",
  },
  {
    label: "LoRA",
    text: "LoRA was proposed by Hu et al. from Microsoft in 2022. The method injects trainable rank decomposition matrices into transformer layers. The paper demonstrated results with a rank of 8, achieving 91.3% accuracy on MNLI.",
  },
  {
    label: "Transformer",
    text: "The paper 'Attention is All You Need' by Vaswani et al. introduced the transformer architecture. The model achieved a BLEU score of 28.4 on WMT 2014 English-to-German. The architecture uses an encoder and decoder each composed of 8 identical layers.",
  },
];

type Verdict = "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE" | string;

type FactResult = {
  fact?: string;
  verdict?: Verdict;
  reasoning?: string;
  evidence_quote?: string;
  evidence_source?: string;
  cove_applied?: boolean;
  cove_meta_verdict?: string | null;
  disputed?: boolean;
  gemini_used?: boolean;
  pdf_used?: boolean;
};

type Correction = {
  fact?: string;
  error_span?: string;
  correction?: string;
  source_url?: string;
  source_sentence?: string;
  changed?: boolean;
};

type VerifyResult = {
  original: string;
  corrected: string;
  facts?: string[];
  results?: FactResult[];
  corrections?: Correction[];
};

type VerifyResponse = {
  success: boolean;
  result: VerifyResult;
  meta: {
    elapsed_ms?: number;
    groq_key_configured?: boolean;
    gemini_key_configured?: boolean;
    timestamp_utc?: string;
  };
};

type Health = {
  status: string;
  groq_key_configured: boolean;
  gemini_key_configured: boolean;
};

type Status = "checking" | "online" | "missing-key" | "offline";

function verdictClass(verdict?: Verdict, disputed?: boolean) {
  if (disputed) return "is-disputed";
  if (verdict === "SUPPORTED") return "is-supported";
  if (verdict === "CONTRADICTED") return "is-contradicted";
  return "is-insufficient";
}

function verdictLabel(verdict?: Verdict, disputed?: boolean) {
  if (disputed) return "DISPUTED";
  return (verdict || "INSUFFICIENT_EVIDENCE").replaceAll("_", " ");
}

function StatTile({
  label,
  value,
  tone,
}: {
  label: string;
  value: number | string;
  tone: "blue" | "green" | "red" | "amber" | "neutral";
}) {
  return (
    <div className={`stat-tile tone-${tone}`}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

function App() {
  const [text, setText] = useState(EXAMPLES[0].text);
  const [verbose, setVerbose] = useState(false);
  const [status, setStatus] = useState<Status>("checking");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<VerifyResponse | null>(null);
  const [error, setError] = useState("");

  async function checkHealth() {
    setStatus("checking");
    try {
      const response = await fetch(`${API_BASE}/health`, {
        signal: AbortSignal.timeout(4000),
      });
      if (!response.ok) throw new Error(response.statusText);
      const payload = (await response.json()) as Health;
      setStatus(payload.groq_key_configured ? "online" : "missing-key");
    } catch {
      setStatus("offline");
    }
  }

  useEffect(() => {
    const initial = window.setTimeout(() => void checkHealth(), 0);
    const timer = window.setInterval(() => void checkHealth(), 30000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, []);

  async function runVerify() {
    const summary = text.trim();
    if (!summary) return;

    setRunning(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE}/api/v1/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ summary, verbose }),
      });

      const payload = (await response.json()) as VerifyResponse | { detail?: string };
      if (!response.ok) {
        throw new Error("detail" in payload ? payload.detail || "Request failed" : "Request failed");
      }

      setResult(payload as VerifyResponse);
      await checkHealth();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Pipeline request failed");
    } finally {
      setRunning(false);
    }
  }

  const summary = useMemo(() => {
    const facts = result?.result.results ?? [];
    const corrections = result?.result.corrections ?? [];
    return {
      total: facts.length,
      supported: facts.filter((fact) => fact.verdict === "SUPPORTED").length,
      contradicted: facts.filter((fact) => fact.verdict === "CONTRADICTED").length,
      disputed: facts.filter((fact) => fact.disputed).length,
      corrections: corrections.length,
    };
  }, [result]);

  const statusCopy = {
    checking: "Checking backend",
    online: "Backend ready",
    "missing-key": "Groq key missing",
    offline: "Backend offline",
  }[status];

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">MetaJudge AI</p>
            <h1>Hallucination Review Console</h1>
          </div>
          <div className={`status-pill status-${status}`}>
            <Pulse size={16} weight="duotone" />
            <span>{statusCopy}</span>
          </div>
        </header>

        <section className="input-panel">
            <div className="panel-heading">
              <div>
                <h2>Summary</h2>
                <p>{API_BASE}</p>
              </div>
              <button className="icon-button" onClick={() => void checkHealth()} title="Refresh backend status">
                <ArrowClockwise size={18} />
              </button>
            </div>

            <textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Paste an AI/ML paper summary"
            />

            <div className="example-row">
              {EXAMPLES.map((example) => (
                <button
                  className="chip-button"
                  key={example.label}
                  onClick={() => setText(example.text)}
                  type="button"
                >
                  <FileText size={15} />
                  {example.label}
                </button>
              ))}
            </div>

            <div className="control-row">
              <label className="toggle-row">
                <input checked={verbose} onChange={(event) => setVerbose(event.target.checked)} type="checkbox" />
                <span>Verbose server logs</span>
              </label>
              <button className="run-button" disabled={running || !text.trim()} onClick={() => void runVerify()}>
                {running ? <ClockCounterClockwise size={18} className="spin" /> : <Play size={18} weight="fill" />}
                {running ? "Running" : "Run Pipeline"}
              </button>
            </div>

            {error ? (
              <div className="alert-row">
                <WarningCircle size={18} />
                <span>{error}</span>
              </div>
            ) : null}
        </section>

        {result ? (
          <section className="results-panel">
            <div className="metrics-grid">
              <StatTile label="Facts" value={summary.total} tone="blue" />
              <StatTile label="Supported" value={summary.supported} tone="green" />
              <StatTile label="Contradicted" value={summary.contradicted} tone="red" />
              <StatTile label="Disputed" value={summary.disputed} tone="amber" />
              <StatTile label="Corrections" value={summary.corrections} tone="neutral" />
            </div>

            <section className="comparison-grid">
              <div className="text-block">
                <h2>Original</h2>
                <p>{result.result.original}</p>
              </div>
              <div className="text-block corrected">
                <h2>Corrected</h2>
                <p>{result.result.corrected}</p>
              </div>
            </section>

            {(result.result.corrections ?? []).length ? (
              <section className="correction-list">
                <h2>Corrections</h2>
                {(result.result.corrections ?? []).map((correction, index) => (
                  <article key={`${correction.fact}-${index}`} className="correction-item">
                    <ClipboardText size={18} />
                    <div>
                      <p>
                        <span className="old-text">{correction.error_span || "unknown"}</span>
                        <span className="arrow-text">-&gt;</span>
                        <span className="new-text">{correction.correction || "unknown"}</span>
                      </p>
                      {correction.source_url ? (
                        <a href={correction.source_url} target="_blank" rel="noreferrer">
                          {correction.source_url}
                        </a>
                      ) : null}
                    </div>
                  </article>
                ))}
              </section>
            ) : null}

            <section className="fact-list">
              <h2>Claim Results</h2>
              {(result.result.results ?? []).map((fact, index) => (
                <article className={`fact-row ${verdictClass(fact.verdict, fact.disputed)}`} key={`${fact.fact}-${index}`}>
                  <div className="fact-icon">
                    {fact.disputed ? (
                      <WarningCircle size={21} />
                    ) : fact.verdict === "SUPPORTED" ? (
                      <CheckCircle size={21} />
                    ) : fact.verdict === "CONTRADICTED" ? (
                      <XCircle size={21} />
                    ) : (
                      <ShieldCheck size={21} />
                    )}
                  </div>
                  <div className="fact-body">
                    <div className="fact-topline">
                      <strong>{fact.fact}</strong>
                      <span>{verdictLabel(fact.verdict, fact.disputed)}</span>
                    </div>
                    {fact.reasoning ? <p>{fact.reasoning}</p> : null}
                    {fact.evidence_quote ? <blockquote>{fact.evidence_quote}</blockquote> : null}
                    <div className="tag-row">
                      {fact.cove_applied ? <span>CoVe: {fact.cove_meta_verdict || "applied"}</span> : null}
                      {fact.gemini_used ? <span>{fact.pdf_used ? "Gemini PDF" : "Gemini"}</span> : null}
                      {fact.evidence_source ? (
                        <a href={fact.evidence_source} target="_blank" rel="noreferrer">
                          Source
                        </a>
                      ) : null}
                    </div>
                  </div>
                </article>
              ))}
            </section>

            <footer className="result-footer">
              <Lightning size={16} />
              <span>{result.meta.elapsed_ms ? `${result.meta.elapsed_ms} ms` : "Complete"}</span>
            </footer>
          </section>
        ) : (
          <section className="empty-panel">
            <ShieldCheck size={24} />
            <span>Ready</span>
          </section>
        )}
      </section>
    </main>
  );
}

export default App;
