"""
streamlit_app.py — Skeptical CoVe-RAG Demo
"""
import streamlit as st
import time, os, sys, re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Skeptical CoVe-RAG", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.main-title{font-size:2rem;font-weight:700;background:linear-gradient(135deg,#7c3aed,#3b82f6);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0}
.sub-title{color:#6b7280;font-size:.88rem;margin-top:2px;margin-bottom:1.5rem}
.step-card{border-radius:10px;padding:14px 16px;margin:6px 0;border:0.5px solid}
.step-1{background:#f5f3ff;border-color:#c4b5fd}
.step-2{background:#eff6ff;border-color:#bfdbfe}
.step-3{background:#f0fdf4;border-color:#bbf7d0}
.step-4{background:#fffbeb;border-color:#fde68a}
.step-4b{background:#fdf4ff;border-color:#e9d5ff}
.step-5{background:#fdf2f8;border-color:#f9a8d4}
.step-6{background:#f0fdfa;border-color:#99f6e4}
.step-label{font-size:.68rem;font-weight:700;letter-spacing:.08em;margin-bottom:4px;text-transform:uppercase}
.step-1 .step-label{color:#7c3aed} .step-2 .step-label{color:#2563eb}
.step-3 .step-label{color:#16a34a} .step-4 .step-label{color:#d97706}
.step-4b .step-label{color:#9333ea} .step-5 .step-label{color:#db2777}
.step-6 .step-label{color:#0d9488}
.step-content{font-size:.84rem;color:#1f2937;line-height:1.6}
.cot-box{background:#fff7ed;border-left:3px solid #f59e0b;padding:8px 12px;
  border-radius:0 6px 6px 0;font-size:.82rem;color:#78350f;font-style:italic;margin:6px 0}
.quote-box{background:#f0fdf4;border-left:3px solid #22c55e;padding:8px 12px;
  border-radius:0 6px 6px 0;font-size:.82rem;color:#14532d;margin:6px 0}
.cove-box{background:#fdf4ff;border-left:3px solid #a855f7;padding:8px 12px;
  border-radius:0 6px 6px 0;font-size:.82rem;color:#581c87;margin:6px 0}
.disputed-box{background:#fff7ed;border-left:3px solid #f97316;padding:8px 12px;
  border-radius:0 6px 6px 0;font-size:.82rem;color:#7c2d12;margin:6px 0}
.badge{display:inline-block;padding:3px 12px;border-radius:12px;font-size:.72rem;font-weight:700}
.badge-supported{background:#d1fae5;color:#065f46}
.badge-contradicted{background:#fee2e2;color:#991b1b}
.badge-insufficient{background:#f3f4f6;color:#6b7280}
.badge-disputed{background:#fff7ed;color:#92400e}
.badge-confirmed{background:#ede9fe;color:#5b21b6}
.badge-overturned{background:#fff7ed;color:#92400e}
.badge-gemini{background:#fdf4ff;color:#7e22ce;font-size:.65rem;margin-left:6px}
del{color:#ef4444;text-decoration:line-through;background:rgba(239,68,68,.08);padding:1px 4px;border-radius:3px}
ins{color:#10b981;text-decoration:none;background:rgba(16,185,129,.1);padding:1px 4px;border-radius:3px;font-weight:600}
.live-fact{background:#1e1b4b;border-radius:10px;padding:16px;font-family:monospace;
  font-size:.8rem;color:#a5b4fc;line-height:2;min-height:120px}
.live-step{color:#86efac}.live-fact-text{color:#e0e7ff;font-weight:600}
.live-done{color:#4ade80}.live-err{color:#f87171}.live-gemini{color:#c084fc}
</style>""", unsafe_allow_html=True)

# Session state
for k,v in [("groq_confirmed",False),("gemini_confirmed",False),
            ("groq_key",os.getenv("GROQ_API_KEY","")),
            ("gemini_key",os.getenv("GEMINI_API_KEY","")),
            ("input_text",""),("result",None),("error",None)]:
    if k not in st.session_state: st.session_state[k] = v

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ API Keys")

    # Groq key
    st.markdown("**Groq** (required)")
    if st.session_state.groq_confirmed:
        st.success("✓ Groq connected")
        if st.button("Change Groq Key", use_container_width=True, key="chg_groq"):
            st.session_state.groq_confirmed = False; st.rerun()
    else:
        gk = st.text_input("Groq key", type="password", placeholder="gsk_...",
                           value=st.session_state.groq_key,
                           label_visibility="collapsed")
        st.caption("Free at console.groq.com")
        if st.button("✓ Confirm Groq Key", type="primary",
                     use_container_width=True, key="conf_groq"):
            if gk.strip():
                st.session_state.groq_key = gk.strip()
                st.session_state.groq_confirmed = True
                os.environ["GROQ_API_KEY"] = gk.strip()
                st.rerun()
            else: st.error("Enter your Groq key first")

    st.markdown("---")

    # Gemini key
    st.markdown("**Gemini Flash** (for second opinion + PDF)")
    if st.session_state.gemini_confirmed:
        st.success("✓ Gemini connected — PDF reading enabled")
        if st.button("Change Gemini Key", use_container_width=True, key="chg_gem"):
            st.session_state.gemini_confirmed = False; st.rerun()
    else:
        gemk = st.text_input("Gemini key", type="password", placeholder="AIza...",
                             value=st.session_state.gemini_key,
                             label_visibility="collapsed")
        st.caption("Free at aistudio.google.com — 1500 req/day")
        if st.button("✓ Confirm Gemini Key", use_container_width=True, key="conf_gem"):
            if gemk.strip():
                st.session_state.gemini_key = gemk.strip()
                st.session_state.gemini_confirmed = True
                os.environ["GEMINI_API_KEY"] = gemk.strip()
                st.rerun()
            else: st.error("Enter your Gemini key first")
        st.info("Without Gemini: INSUFFICIENT facts show as DISPUTED.\nWith Gemini: full PDF analysis for deeper verification.")

    st.markdown("---")
    st.markdown("### 🎛️ Options")
    show_reasoning = st.toggle("Judge reasoning",   value=True)
    show_cove      = st.toggle("CoVe verification", value=True)
    show_raw       = st.toggle("Raw JSON",          value=False)

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""**Skeptical CoVe-RAG**

6-step hallucination correction:
1. Atomic decomposition
2. Adversarial queries
3. Hybrid retrieval
4. LLM judge (Groq 70B)
4b. ★ Gemini second opinion
5. CoVe meta-verification
6. Surgical editor""")

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">⚡ Skeptical CoVe-RAG</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Meta-Verification of LLM Judges through Adversarial Falsification for Hallucination Correction</p>', unsafe_allow_html=True)

if not st.session_state.groq_confirmed:
    st.info("👈 Enter and confirm your Groq API key in the sidebar to begin.")
    st.stop()

os.environ["GROQ_API_KEY"]   = st.session_state.groq_key
os.environ["GEMINI_API_KEY"] = st.session_state.gemini_key

# Reinitialise Groq clients in modules
try:
    import groq as _groq
    _nc = _groq.Groq(api_key=st.session_state.groq_key)
    for mn in ["modules.atomicizer","modules.query_generator","modules.judge","modules.cove_loop","modules.editor"]:
        if mn in sys.modules:
            m = sys.modules[mn]
            if hasattr(m,"client"): m.client = _nc
except Exception: pass

# ── Examples ───────────────────────────────────────────────────────────────
EXAMPLES = {
    "BERT — metric error":      "BERT, introduced by Google in 2018, uses a bidirectional transformer encoder. It was pre-trained on BookCorpus and English Wikipedia, and achieved 80.5% F1 on the SQuAD 2.0 benchmark. The paper was authored by Devlin et al.",
    "FActScore — author error": "FActScore was proposed by Lee et al. in 2022 as a framework for evaluating factual precision. It decomposes model outputs into atomic facts and evaluates each independently against a knowledge source.",
    "LoRA — multiple errors":   "LoRA was proposed by Hu et al. from Microsoft in 2022. The method injects trainable rank decomposition matrices into transformer layers. The paper demonstrated results with a rank of 8, achieving 91.3% accuracy on MNLI.",
    "Transformer — layer error":"The paper 'Attention is All You Need' by Vaswani et al. introduced the transformer architecture. The model achieved a BLEU score of 28.4 on WMT 2014 English-to-German. The architecture uses an encoder and decoder each composed of 8 identical layers.",
}

st.markdown("### 📝 Input Summary")
cin, cex = st.columns([3,1])
with cex:
    st.markdown("**Examples:**")
    for lbl in EXAMPLES:
        if st.button(lbl, use_container_width=True, key=f"ex_{lbl}"):
            st.session_state.input_text = EXAMPLES[lbl]
            st.session_state.result = None
            st.session_state.error  = None
            st.rerun()
with cin:
    input_text = st.text_area("Summary", value=st.session_state.input_text,
        height=140, placeholder="Paste any AI-generated summary about an AI/ML paper...",
        label_visibility="collapsed")

rc,cc,_ = st.columns([1,1,5])
with rc:
    run_clicked = st.button("▶  Run Pipeline", type="primary",
        use_container_width=True, disabled=not input_text.strip())
with cc:
    if st.button("✕ Clear", use_container_width=True):
        st.session_state.input_text=""
        st.session_state.result=None
        st.session_state.error=None
        st.rerun()

# ── Run ────────────────────────────────────────────────────────────────────
if run_clicked and input_text.strip():
    st.session_state.input_text = input_text
    st.session_state.result = None
    st.session_state.error  = None

    # Force-set both keys in environment RIGHT NOW before any module call
    os.environ["GROQ_API_KEY"]   = st.session_state.groq_key
    os.environ["GEMINI_API_KEY"] = st.session_state.gemini_key

    # Re-init Groq clients in all modules with fresh key
    try:
        import groq as _g
        _nc = _g.Groq(api_key=st.session_state.groq_key)
        for _mn in ["modules.atomicizer","modules.query_generator",
                    "modules.judge","modules.cove_loop","modules.editor"]:
            if _mn in sys.modules and hasattr(sys.modules[_mn],"client"):
                sys.modules[_mn].client = _nc
    except Exception: pass

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Running")
    overall_prog  = st.progress(0)
    overall_label = st.empty()
    console_ph    = st.empty()
    log_lines     = []

    def log(msg, kind="info"):
        css={"step":"live-step","fact":"live-fact-text","done":"live-done",
             "err":"live-err","gemini":"live-gemini"}.get(kind,"")
        span=f'<span class="{css}">{msg}</span>' if css else msg
        log_lines.append(span)
        visible=log_lines[-20:]
        console_ph.markdown(
            f'<div class="live-fact">{"<br>".join(visible)}</div>',
            unsafe_allow_html=True)

    try:
        from config import MAX_FACTS
        from modules.atomicizer       import atomicize
        from modules.query_generator  import generate_skeptical_queries
        from modules.retriever        import retrieve_evidence, format_evidence_block, KNOWN_PAPERS
        from modules.judge            import judge_claim, VERDICT_CONTRADICTED, VERDICT_INSUFFICIENT
        from modules.cove_loop        import run_cove_verification
        from modules.editor           import edit_sentence, apply_corrections_to_summary
        from modules.deep_verifier    import deep_verify

        summary = input_text.strip()

        overall_label.markdown("**Step 1 of 6 — Atomicizing...**")
        overall_prog.progress(0.05)
        log("▶ Step 1: Atomicizing summary...", "step")
        facts = atomicize(summary)[:MAX_FACTS]
        log(f"  → {len(facts)} atomic facts extracted", "done")

        all_results, corrections = [], []
        n = len(facts)

        for i, fact in enumerate(facts):
            overall_prog.progress(0.1 + (i/n)*0.85)
            overall_label.markdown(f"**Processing fact {i+1} of {n}...**")

            log(f"", "info")
            log(f"── Fact {i+1}/{n} ─────────────────────────", "info")
            log(f"  {fact[:80]}{'...' if len(fact)>80 else ''}", "fact")

            # Step 2
            log("▶ Step 2: Generating adversarial queries...", "step")
            queries = generate_skeptical_queries(fact)
            for q in queries: log(f"  ↯ {q[:75]}", "info")

            # Step 3
            log("▶ Step 3: Retrieving evidence...", "step")
            evidence       = retrieve_evidence(queries, fact=fact, context=summary)
            evidence_block = format_evidence_block(evidence)
            log(f"  → {len(evidence)} evidence items (arXiv direct + adversarial + web)", "done")

            # Step 4
            log("▶ Step 4: Groq 70B judging claim...", "step")
            judge_result = judge_claim(fact, evidence_block)
            verdict_raw  = judge_result["verdict"]
            log(f"  → Groq verdict: {verdict_raw}",
                "done" if verdict_raw == "SUPPORTED" else
                "err"  if verdict_raw == "CONTRADICTED" else "info")
            if judge_result.get("reasoning"):
                log(f"  → {judge_result['reasoning'][:90]}", "info")

            # Step 4b — Deep verification: LangChain + Gemini (key passed explicitly)
            if verdict_raw == VERDICT_INSUFFICIENT:
                gemini_key = st.session_state.gemini_key if st.session_state.gemini_confirmed else ""
                has_gemini = bool(gemini_key)
                log_msg = "▶ Step 4b: ★ Deep verify (LangChain arXiv + Papers With Code" + (" + Gemini PDF)" if has_gemini else ") — no Gemini key") + "..."
                log(log_msg, "gemini" if has_gemini else "step")
                try:
                    from modules.deep_verifier import deep_verify
                    second = deep_verify(
                        fact, summary, evidence_block,
                        KNOWN_PAPERS,
                        gemini_key=gemini_key,
                        verbose=False,
                    )
                    if second and second.get("verdict") != VERDICT_INSUFFICIENT:
                        judge_result["verdict"]         = second.get("verdict", VERDICT_INSUFFICIENT)
                        judge_result["reasoning"]       = second.get("reasoning","")
                        judge_result["evidence_quote"]  = second.get("evidence_quote","")
                        judge_result["evidence_source"] = second.get("evidence_source","")
                        judge_result["gemini_used"]     = second.get("gemini_used", False)
                        judge_result["pdf_used"]        = second.get("pdf_used", False)
                        verdict_raw = judge_result["verdict"]
                        src_tag = " (PDF)" if second.get("pdf_used") else " (Gemini)" if second.get("gemini_used") else " (enriched search)"
                        log(f"  → Deep verify{src_tag}: {verdict_raw}",
                            "gemini" if verdict_raw=="CONTRADICTED" else "done")
                    else:
                        judge_result["disputed"] = True
                        # Mark gemini_used even if verdict was INSUFFICIENT
                        # so UI shows "Gemini tried but couldn't verify" not "add key"
                        if has_gemini:
                            judge_result["gemini_used"] = True
                            judge_result["pdf_used"]    = second.get("pdf_used", False) if second else False
                        log("  → Gemini tried but could not verify — marked DISPUTED", "gemini" if has_gemini else "info")
                except Exception as e:
                    judge_result["disputed"] = True
                    if has_gemini:
                        judge_result["gemini_used"] = True
                    log(f"  → Deep verify error: {e}", "err")

            final = judge_result

            # Step 5
            if final["verdict"] == VERDICT_CONTRADICTED:
                log("▶ Step 5: ★ CoVe meta-verification...", "step")
                final = run_cove_verification(fact, judge_result, evidence_block)
                final["fact"] = fact
                meta = final.get("cove_meta_verdict","")
                log(f"  → CoVe: {meta}",
                    "done" if meta=="CONFIRMED_CONTRADICTION" else "err")
            else:
                final["cove_applied"]      = False
                final["cove_meta_verdict"] = None
                final["fact"]              = fact
                log("▶ Step 5: CoVe skipped", "step")

            all_results.append(final)

            # Step 6
            if (final["verdict"]==VERDICT_CONTRADICTED and
                    final.get("cove_meta_verdict")=="CONFIRMED_CONTRADICTION"):
                log("▶ Step 6: Applying surgical correction...", "step")
                sents  = [s.strip() for s in summary.replace(".\n",". ").split(". ") if s.strip()]
                fw     = set(fact.lower().split())
                src    = max(sents, key=lambda s: len(set(s.lower().split())&fw), default=summary)
                src    = src + ("." if not src.endswith(".") else "")
                edit   = edit_sentence(src, fact, final, evidence_block)
                if edit["changed"]:
                    corrections.append({"fact":fact,"source_sentence":src,**edit})
                    log(f"  ✓ Fixed: '{edit['error_span']}' → '{edit['correction']}'","done")
                else:
                    log("  ~ Could not extract exact correction span","info")
            else:
                log("▶ Step 6: Editor skipped","step")

            time.sleep(0.1)

        corrected = apply_corrections_to_summary(summary, corrections)
        overall_prog.progress(1.0)
        overall_label.markdown("**✅ Pipeline complete!**")
        log("","info")
        n_contra   = sum(1 for r in all_results if r["verdict"]=="CONTRADICTED")
        n_disputed = sum(1 for r in all_results if r.get("disputed"))
        log(f"✅ DONE — {len(corrections)} correction(s) | {n_contra} contradiction(s) | {n_disputed} disputed","done")
        time.sleep(0.5)

        st.session_state.result = {
            "original":    summary,
            "corrected":   corrected,
            "results":     all_results,
            "corrections": corrections,
        }

    except Exception as e:
        import traceback
        st.session_state.error = f"{e}\n\n{traceback.format_exc()}"
        log(f"✗ ERROR: {e}", "err")

# ── Error ──────────────────────────────────────────────────────────────────
if st.session_state.error:
    st.error("Pipeline failed")
    with st.expander("Error details"):
        st.code(st.session_state.error)

# ── Results ────────────────────────────────────────────────────────────────
if st.session_state.result:
    r           = st.session_state.result
    facts_res   = r["results"]
    corrections = r["corrections"]
    n_contra    = sum(1 for x in facts_res if x["verdict"]=="CONTRADICTED")
    n_cove_rev  = sum(1 for x in facts_res if x.get("cove_applied") and x.get("cove_meta_verdict")=="OVERTURNED")
    n_fixed     = len(corrections)
    n_disputed  = sum(1 for x in facts_res if x.get("disputed"))
    n_gemini    = sum(1 for x in facts_res if x.get("gemini_used"))

    st.markdown("---")
    st.markdown("### 📊 Results")

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,val,lbl,color in [
        (c1,len(facts_res),  "Facts checked",      "#7c3aed"),
        (c2,n_contra,        "Contradictions",      "#ef4444"),
        (c3,n_disputed,      "Disputed/Unverified", "#f97316"),
        (c4,n_fixed,         "Corrections applied", "#10b981"),
        (c5,n_gemini,        "Gemini analysed",     "#9333ea"),
    ]:
        with col:
            st.markdown(
                f'<div style="background:#f9fafb;border:0.5px solid #e5e7eb;border-radius:10px;'
                f'padding:12px;text-align:center">'
                f'<div style="font-size:1.8rem;font-weight:700;color:{color}">{val}</div>'
                f'<div style="font-size:.7rem;color:#9ca3af;margin-top:3px">{lbl}</div>'
                f'</div>', unsafe_allow_html=True)

    # Output diff
    st.markdown("### ✏️ Corrected Output")
    orig = r["original"]; corr = r["corrected"]

    if orig == corr:
        st.success("No corrections applied.")
        st.info(corr)
    else:
        oc,cc2 = st.columns(2)
        with oc:
            st.markdown("**Original**")
            st.markdown(f'<div style="background:#fef2f2;padding:14px;border-radius:8px;border:1px solid #fecaca;font-size:.88rem;line-height:1.8">{orig}</div>', unsafe_allow_html=True)
        with cc2:
            st.markdown("**Corrected**")
            h = corr
            for c in corrections:
                if c.get("correction"):
                    h = h.replace(c["correction"], f'<ins>{c["correction"]}</ins>',1)
            st.markdown(f'<div style="background:#f0fdf4;padding:14px;border-radius:8px;border:1px solid #bbf7d0;font-size:.88rem;line-height:1.8">{h}</div>', unsafe_allow_html=True)
        if corrections:
            st.markdown("**Changes:**")
            for c in corrections:
                src = c.get("source_url","")
                sl  = f' <a href="{src}" target="_blank" style="font-size:.75rem;color:#6366f1">↗ source</a>' if src else ""
                st.markdown(f'- <del>{c.get("error_span","?")}</del> → <ins>{c.get("correction","?")}</ins>{sl}', unsafe_allow_html=True)

    # Disputed summary
    disputed_facts = [x for x in facts_res if x.get("disputed")]
    if disputed_facts:
        st.markdown("### ⚠️ Disputed / Unverified Facts")
        st.caption("These facts could not be verified or contradicted with available evidence. They may be correct or incorrect — further manual verification is recommended.")
        for x in disputed_facts:
            gemini_tag = " (Gemini also couldn't verify)" if x.get("gemini_used") else " (add Gemini key for deeper analysis)"
            st.markdown(
                f'<div class="step-card" style="background:#fff7ed;border-color:rgba(249,115,22,.25)">'
                f'<div style="font-size:.7rem;font-weight:700;color:#ea580c;margin-bottom:4px">DISPUTED{gemini_tag}</div>'
                f'<div style="font-size:.85rem;color:#1f2937">{x.get("fact","")}</div>'
                f'</div>', unsafe_allow_html=True)

    # Chain of Thought
    st.markdown("### 🧠 Chain of Thought — Full Reasoning per Fact")
    st.caption("Complete 6-step reasoning for every fact. Contradicted and disputed facts auto-expand.")

    for i, res in enumerate(facts_res):
        verdict   = res.get("verdict","INSUFFICIENT_EVIDENCE")
        fact      = res.get("fact","")
        reasoning = res.get("reasoning","")
        quote     = res.get("evidence_quote","")
        src       = res.get("evidence_source","")
        cove_app  = res.get("cove_applied",False)
        cove_meta = res.get("cove_meta_verdict","")
        disputed  = res.get("disputed",False)
        gem_used  = res.get("gemini_used",False)
        pdf_used  = res.get("pdf_used",False)

        icon = {"SUPPORTED":"✅","CONTRADICTED":"❌","INSUFFICIENT_EVIDENCE":"⚪"}.get(verdict,"⚪")
        if disputed: icon = "⚠️"

        badge_cls = {"SUPPORTED":"badge-supported","CONTRADICTED":"badge-contradicted",
                     "INSUFFICIENT_EVIDENCE":"badge-insufficient"}.get(verdict,"badge-insufficient")
        if disputed: badge_cls = "badge-disputed"

        tags = ""
        if disputed:    tags += " ⚠️ Disputed"
        if gem_used:    tags += " 🔮 Gemini" + (" + PDF" if pdf_used else "")
        if cove_app:    tags += " 🔄 CoVe reversed" if cove_meta=="OVERTURNED" else " ★ CoVe confirmed"

        expanded = verdict=="CONTRADICTED" or disputed

        with st.expander(f"{icon} Fact {i+1}: {fact[:70]}{'...' if len(fact)>70 else ''}{tags}", expanded=expanded):
            verdict_display = "DISPUTED" if disputed else verdict.replace("_"," ")
            st.markdown(f'<span class="badge {badge_cls}">{verdict_display}</span>'
                        + (f'<span class="badge badge-gemini">Gemini{"+ PDF" if pdf_used else ""}</span>' if gem_used else ""),
                        unsafe_allow_html=True)
            st.markdown(f"**Claim:** {fact}")
            st.markdown("")

            # Steps 1-3
            for step_n, step_cls, step_lbl, step_desc in [
                ("1","step-1","Atomicizer","Self-contained verifiable claim — explicit subject, one claim only."),
                ("2","step-2","Adversarial query generator","Conflict-seeking queries to disprove rather than confirm."),
                ("3","step-3","Hybrid retrieval","arXiv Direct (authoritative) + arXiv Adversarial + DuckDuckGo web."),
            ]:
                st.markdown(f'<div class="step-card {step_cls}"><div class="step-label">Step {step_n} — {step_lbl}</div><div class="step-content">{step_desc}</div></div>', unsafe_allow_html=True)

            # Step 4
            if show_reasoning:
                cot  = f'<div class="cot-box">💭 {reasoning}</div>' if reasoning else ""
                qbox = ""
                if quote:
                    sl = f'<br><a href="{src}" target="_blank" style="font-size:.73rem;color:#6366f1">↗ {src[:60]}</a>' if src else ""
                    qbox = f'<div class="quote-box">📌 "{quote}"{sl}</div>'
                st.markdown(f'<div class="step-card step-4"><div class="step-label">Step 4 — Groq 70B judge (chain of thought)</div><div class="step-content">{cot}{qbox}</div></div>', unsafe_allow_html=True)

            # Step 4b — Gemini / disputed
            if gem_used or disputed:
                if disputed and not gem_used:
                    s4b = '<div class="disputed-box">⚠️ This fact could not be verified. Add your <strong>Gemini API key</strong> in the sidebar — Gemini will read the full paper PDF including results tables to find the actual value.</div>'
                elif disputed and gem_used:
                    pdf_note = " (read full PDF + results tables)" if pdf_used else " (text-only — PDF unavailable)"
                    s4b = f'<div class="disputed-box">⚠️ Gemini Flash{pdf_note} searched the full paper but could not find a definitive value to confirm or contradict this claim. The specific value may be in supplementary materials not available via arXiv. <strong>Manual verification recommended.</strong></div>'
                else:
                    pdf_note = " — read full paper PDF" if pdf_used else " — text analysis"
                    s4b = f'<div class="cove-box">🔮 Gemini Flash{pdf_note}: {reasoning[:150] if reasoning else "Verified claim"}. Verdict: <strong>{verdict}</strong></div>'
                st.markdown(f'<div class="step-card step-4b"><div class="step-label">Step 4b — Gemini Flash second opinion</div><div class="step-content">{s4b}</div></div>', unsafe_allow_html=True)

            # Step 5 CoVe
            if show_cove:
                if cove_app:
                    if cove_meta=="CONFIRMED_CONTRADICTION":
                        cv = '<span class="badge badge-confirmed">★ Confirmed</span><br><div class="cove-box">CoVe verified the judge\'s quote exists in evidence and contradicts the claim. Correction authorised.</div>'
                    else:
                        cv = '<span class="badge badge-overturned">🔄 Overturned</span><br><div class="cove-box">CoVe could not verify the evidence quote. Verdict reversed — no false correction applied.</div>'
                else:
                    cv = '<span style="color:#9ca3af;font-size:.8rem">Not triggered (verdict was not CONTRADICTED)</span>'
                st.markdown(f'<div class="step-card step-5"><div class="step-label">Step 5 — CoVe meta-verification ★</div><div class="step-content">{cv}</div></div>', unsafe_allow_html=True)

            # Step 6
            cm = next((c for c in corrections if c.get("fact")==fact or c.get("error_span","") in fact), None)
            if cm:
                sl = f'<br><a href="{cm.get("source_url","#")}" target="_blank" style="font-size:.73rem;color:#6366f1">↗ source</a>' if cm.get("source_url") else ""
                s6 = f'Surgical correction: <del>{cm.get("error_span","?")}</del> → <ins>{cm.get("correction","?")}</ins>{sl}'
            elif verdict=="CONTRADICTED":
                s6 = "Contradiction confirmed but exact correction span could not be extracted."
            elif disputed:
                s6 = '<span style="color:#f97316">No correction applied — fact is disputed, not confirmed as wrong.</span>'
            else:
                s6 = '<span style="color:#9ca3af">No correction needed.</span>'
            st.markdown(f'<div class="step-card step-6"><div class="step-label">Step 6 — Targeted RARR editor</div><div class="step-content">{s6}</div></div>', unsafe_allow_html=True)

    if show_raw:
        with st.expander("🗂 Raw JSON"):
            st.json({"original":r["original"],"corrected":r["corrected"],"corrections":r["corrections"]})

elif not run_clicked and not st.session_state.result:
    st.markdown("---")
    st.markdown("""<div style="text-align:center;padding:3rem;color:#9ca3af">
      <div style="font-size:3rem">⚡</div>
      <div style="font-size:1.1rem;font-weight:500;margin-top:1rem;color:#374151">Ready to fact-check</div>
      <div style="font-size:.9rem;margin-top:.5rem">Pick an example or paste a summary, then click Run Pipeline</div>
    </div>""", unsafe_allow_html=True)