"""
KGMS Streamlit Application
Run: streamlit run app.py  (from the kgms/ directory)
  or: python app.py         (auto-relaunches with streamlit)

Three views:
  1. Phase A — paper input + gap analysis + candidate papers
  2. PDF Collector — which gaps have PDFs, which need them
  3. Phase B — generated learning document
"""

import os
import sys
import json
import tempfile
import warnings

warnings.filterwarnings("ignore")


# ── Auto-relaunch guard ────────────────────────────────────────────────────
# When run as `python app.py`, Streamlit context is missing.
# Detect this and re-exec with `streamlit run` automatically.
def _is_streamlit_context() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if not _is_streamlit_context():
    import subprocess

    print("Launching Streamlit…  (use Ctrl+C to stop)")
    result = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", __file__] + sys.argv[1:],
        cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
    )
    sys.exit(result.returncode)
# ── End guard ──────────────────────────────────────────────────────────────

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Knowledge Gap Mitigation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from core.models import PipelineState
from utils.cache import cache_stats, clear_cache
from utils.llm import RateLimitError


# ── Session state helpers ──────────────────────────────────────────────────


def _init_state():
    defaults = {
        "phase": "input",  # input | phase_a | pdf_collect | phase_b
        "pipeline_state": None,
        "disabled_gaps": set(),
        "custom_gaps": [],
        "depth": 0,
        "uploaded_pdfs": {},  # paper_id -> tmp file path
        "log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def _log(msg: str):
    st.session_state.log.append(msg)


def _has_api_key() -> bool:
    """Return True if at least one LLM API key is available for the active provider."""
    provider = os.environ.get("LLM_PROVIDER", "auto").lower()
    if provider == "groq":
        return bool(os.environ.get("GROQ_API_KEY"))
    if provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    # auto: either key is fine
    return bool(os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def _rate_limit_banner(e) -> str:
    """Build the st.error message for a RateLimitError."""
    provider = getattr(e, "provider", "") or "groq"
    billing_urls = {
        "groq": "https://console.groq.com/settings/billing",
        "openai": "https://platform.openai.com/settings/organization/billing",
    }
    billing_url = billing_urls.get(provider, billing_urls["groq"])
    billing_label = f"{provider.capitalize()} billing"
    provider_label = provider.capitalize()
    return (
        f"🚫 **{provider_label} Rate Limit Reached**\n\n"
        f"The quota for **{e.model}** has been exhausted.\n\n"
        + (
            f"⏳ Please try again in **{e.wait_time}**."
            if e.wait_time
            else "Please check your quota / billing."
        )
        + f"\n\nUpgrade at [{billing_label}]({billing_url})."
    )


# ── Sidebar ────────────────────────────────────────────────────────────────


def render_sidebar():
    with st.sidebar:
        st.title("📚 KGMS")
        st.caption("Knowledge Gap Mitigation System")
        st.divider()

        phase = st.session_state.phase
        steps = [
            ("input", "1. Enter Paper"),
            ("phase_a", "2. Gap Analysis"),
            ("pdf_collect", "3. Collect PDFs"),
            ("phase_b", "4. Learning Document"),
        ]
        for step_id, label in steps:
            is_active = step_id == phase
            prefix = "▶ " if is_active else "  "
            st.markdown(
                f"{'**' if is_active else ''}{prefix}{label}{'**' if is_active else ''}"
            )

        st.divider()

        # ── LLM Provider ───────────────────────────────────────────────────
        provider_choice = st.selectbox(
            "LLM Provider",
            options=["auto", "groq", "openai"],
            index=["auto", "groq", "openai"].index(
                os.environ.get("LLM_PROVIDER", "auto").lower()
            ),
            format_func=lambda x: {
                "auto": "🤖 Auto (Groq → OpenAI)",
                "groq": "⚡ Groq (free, fast)",
                "openai": "🟢 OpenAI (GPT-4o)",
            }[x],
            help="auto = use Groq if GROQ_API_KEY is set, else OpenAI",
        )
        os.environ["LLM_PROVIDER"] = provider_choice

        # Show relevant API key inputs
        if provider_choice in ("groq", "auto"):
            groq_key = st.text_input(
                "Groq API Key",
                value=os.environ.get("GROQ_API_KEY", ""),
                type="password",
                help="Free key at https://console.groq.com",
            )
            if groq_key:
                os.environ["GROQ_API_KEY"] = groq_key

        if provider_choice in ("openai", "auto"):
            oai_key = st.text_input(
                "OpenAI API Key",
                value=os.environ.get("OPENAI_API_KEY", ""),
                type="password",
                help="Key at https://platform.openai.com/api-keys",
            )
            if oai_key:
                os.environ["OPENAI_API_KEY"] = oai_key

        # Cache stats
        stats = cache_stats()
        st.caption(f"Cache: {stats['entries']} entries ({stats['size_kb']} KB)")
        if st.button("Clear cache", use_container_width=True):
            clear_cache()
            st.success("Cache cleared")

        # Reset
        st.divider()
        if st.button("🔄 Start over", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ── Phase: Input ───────────────────────────────────────────────────────────


def render_input():
    st.title("📚 Knowledge Gap Mitigation System")
    st.subheader("Understand any research paper — automatically.")
    st.markdown(
        "This system analyses a research paper, identifies concepts you need to "
        "know before reading it, and generates a personalised learning document "
        "sourced from the paper's own reference hierarchy."
    )
    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        paper_input = st.text_input(
            "Paper identifier",
            placeholder="arXiv ID (e.g. 2404.16130), DOI, or Semantic Scholar ID",
            help="Examples: 2404.16130 | 10.1109/ICCIT57492.2022.10103286",
        )

        base_pdf_upload = st.file_uploader(
            "Upload Base Paper PDF (Optional)",
            type=["pdf"],
            help="Providing the PDF improves gap detection compared to using just the abstract.",
        )

        depth_input = st.selectbox(
            "Reference Depth",
            options=[0, 1, 2, 3],
            index=1,
            format_func=lambda x: f"{x} - "
            + [
                "Gap detection only",
                "Direct references (speed:fast)",
                "Refs-of-refs (speed:medium)",
                "Full graph (speed:slow)",
            ][x],
            help="How deep to search the reference graph.",
        )

        st.markdown("**Optional: specify concepts you want explained**")
        custom_gap_input = st.text_area(
            "Custom concepts (one per line)",
            placeholder="e.g.\nattention mechanism\ngraph neural network",
            height=100,
        )

    with col2:
        st.markdown("**Examples to try:**")
        examples = [
            ("2404.16130", "GraphRAG paper"),
            ("2309.15217", "RAGAS paper"),
            ("arXiv:2405.20139", "GNN-RAG paper"),
        ]
        for ex_id, ex_label in examples:
            if st.button(f"📄 {ex_label}", use_container_width=True, key=f"ex_{ex_id}"):
                if not _has_api_key():
                    st.warning(
                        "⚠️ Please set an API key (Groq or OpenAI) in the sidebar or .env file before proceeding."
                    )
                else:
                    user_gaps = [
                        line.strip()
                        for line in custom_gap_input.splitlines()
                        if line.strip()
                    ]
                    st.session_state.custom_gaps = user_gaps
                    st.session_state.depth = depth_input
                    st.session_state._paper_input = ex_id
                    st.session_state.ba_pdf_path = None
                    st.session_state.phase = "phase_a"
                    st.rerun()

    if st.button(
        "🔍 Analyse Paper",
        type="primary",
        use_container_width=True,
        disabled=not paper_input.strip(),
    ):
        if not _has_api_key():
            st.warning(
                "⚠️ Please set an API key (Groq or OpenAI) in the sidebar or .env file before proceeding."
            )
        else:
            user_gaps = [
                line.strip() for line in custom_gap_input.splitlines() if line.strip()
            ]
            st.session_state.custom_gaps = user_gaps
            st.session_state.depth = depth_input
            st.session_state._paper_input = paper_input.strip()

            if base_pdf_upload:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(base_pdf_upload.read())
                tmp.close()
                st.session_state.ba_pdf_path = tmp.name
            else:
                st.session_state.ba_pdf_path = None

            st.session_state.phase = "phase_a"
            st.rerun()


# ── Phase: Phase A ─────────────────────────────────────────────────────────


def render_phase_a():
    st.title("🔍 Gap Analysis")
    paper_input = st.session_state.get("_paper_input", "")
    user_gaps = st.session_state.get("custom_gaps", [])
    depth = st.session_state.get("depth", 1)
    ba_pdf_path = st.session_state.get("ba_pdf_path", None)

    if not paper_input:
        st.error("No paper specified. Go back and enter a paper ID.")
        return

    # Run Phase A if not yet done
    if st.session_state.pipeline_state is None:
        progress_area = st.empty()
        log_expander = st.expander("Terminal Logs", expanded=True)
        log_placeholder = log_expander.empty()

        def cb(msg):
            _log(msg)
            progress_area.info(f"⏳ {msg}")
            log_placeholder.code("\n".join(st.session_state.log[-20:]), language="text")

        with st.spinner("Running Phase A analysis…"):
            from pipeline import run_phase_a

            try:
                state = run_phase_a(
                    paper_input=paper_input,
                    user_gaps=user_gaps,
                    reference_depth=depth,
                    pdf_path=ba_pdf_path,
                    progress_callback=cb,
                )
                st.session_state.pipeline_state = state
            except RateLimitError as e:
                st.error(_rate_limit_banner(e))
                st.session_state.phase = "input"
                return
            except Exception as e:
                st.error(f"Phase A failed: {e}")
                return

        progress_area.success("✅ Phase A complete!")

    state: PipelineState = st.session_state.pipeline_state

    if state.errors:
        for err in state.errors:
            st.error(err)
        return

    # ── Paper info ─────────────────────────────────────────────────────────
    if state.ba_paper:
        with st.expander("📄 Base Article", expanded=False):
            st.markdown(f"**{state.ba_paper.title}** ({state.ba_paper.year})")
            st.markdown(f"*{state.ba_paper.abstract[:500]}…*")

    # ── Graph stats ────────────────────────────────────────────────────────
    if state.reference_graph and state.all_papers:
        from phase_a.graph import graph_summary

        summary = graph_summary(state.reference_graph, state.all_papers)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Papers", summary["total_papers"])
        c2.metric("🏛 Foundation", summary["foundation"])
        c3.metric("📈 Development", summary["development"])
        c4.metric("🚀 Frontier", summary["frontier"])

    st.divider()

    # ── Gap cards ──────────────────────────────────────────────────────────
    st.subheader(f"Detected Knowledge Gaps ({len(state.gaps)})")
    st.caption(
        "Toggle gaps off if you already know them. "
        "The system will skip them in Phase B."
    )

    disabled = st.session_state.disabled_gaps

    for gap in state.gaps:
        conf_color = (
            "🟢" if gap.confidence > 0.75 else ("🟡" if gap.confidence > 0.5 else "🔴")
        )
        diff_badge = {"beginner": "🌱", "intermediate": "🌿", "advanced": "🌳"}.get(
            gap.difficulty.value, ""
        )
        type_badge = {
            "terminology": "📖",
            "methodology": "⚙️",
            "benchmark": "📊",
            "historical": "📜",
            "mathematical": "🔢",
        }.get(gap.gap_type.value, "")

        col_toggle, col_content = st.columns([0.05, 0.95])
        with col_toggle:
            enabled = st.checkbox(
                "",
                value=(gap.gap_id not in disabled),
                key=f"gap_toggle_{gap.gap_id}",
                label_visibility="collapsed",
            )
            if enabled:
                disabled.discard(gap.gap_id)
            else:
                disabled.add(gap.gap_id)

        with col_content:
            with st.expander(
                f"{conf_color} {type_badge} {diff_badge}  **{gap.concept}**  "
                f"— *{gap.gap_type.value}* | confidence: {int(gap.confidence * 100)}%",
                expanded=False,
            ):
                st.markdown(f"**Why needed:** {gap.why_needed}")
                if gap.source_passage:
                    st.markdown(f'**Source passage:** *"{gap.source_passage[:200]}"*')
                st.markdown(
                    f"**Domain:** {gap.domain} | **Difficulty:** {gap.difficulty.value}"
                )

                # Show candidates for this gap
                gap_candidates = [c for c in state.candidates if c.gap_id == gap.gap_id]
                if gap_candidates:
                    st.markdown("**Candidate papers:**")
                    for c in gap_candidates[:3]:
                        pdf_icon = "✅" if c.pdf_available else "❌"
                        st.markdown(
                            f"- {pdf_icon} **{c.paper.title[:70]}** ({c.paper.year}) "
                            f"— Layer: {c.paper.layer.value} "
                            f"| Score: {c.relevance_score:.3f}"
                        )
                        st.caption(f"  *{c.rationale[:200]}*")

    # ── Navigation ────────────────────────────────────────────────────────
    st.divider()
    active_gaps = [g for g in state.gaps if g.gap_id not in disabled]
    st.info(f"{len(active_gaps)} gaps active, {len(disabled)} disabled")

    # Export buttons for candidate papers
    if state.candidates:
        from phase_a.candidates import candidates_to_bibtex, candidates_to_csv

        active_candidates = [c for c in state.candidates if c.gap_id not in disabled]
        col_bib, col_csv, col_next = st.columns([1, 1, 2])
        with col_bib:
            bib = candidates_to_bibtex(active_candidates)
            st.download_button(
                "⬇ BibTeX",
                data=bib,
                file_name=f"kgms_candidates_{state.ba_paper.paper_id[:10]}.bib",
                mime="text/plain",
                help="Import into Zotero or Mendeley before collecting PDFs",
                use_container_width=True,
            )
        with col_csv:
            csv = candidates_to_csv(active_candidates)
            st.download_button(
                "⬇ CSV",
                data=csv,
                file_name=f"kgms_candidates_{state.ba_paper.paper_id[:10]}.csv",
                mime="text/csv",
                help="Track PDF collection status in a spreadsheet",
                use_container_width=True,
            )
        with col_next:
            if st.button(
                "➡ Continue to PDF Collection", type="primary", use_container_width=True
            ):
                st.session_state.phase = "pdf_collect"
                st.rerun()
    else:
        if st.button("➡ Continue to PDF Collection", type="primary"):
            st.session_state.phase = "pdf_collect"
            st.rerun()


# ── Phase: PDF Collector ───────────────────────────────────────────────────


def render_pdf_collect():
    st.title("📁 PDF Collection")
    st.markdown(
        "The system will automatically try to fetch open-access PDFs. "
        "You can also upload PDFs manually for papers behind paywalls."
    )

    state: PipelineState = st.session_state.pipeline_state
    if not state:
        st.error("Run Phase A first.")
        return

    disabled = st.session_state.disabled_gaps
    active_candidates = [c for c in state.candidates if c.gap_id not in disabled]

    # Deduplicate by paper_id
    seen_pids: set = set()
    unique_candidates = []
    for c in active_candidates:
        if c.paper.paper_id not in seen_pids:
            seen_pids.add(c.paper.paper_id)
            unique_candidates.append(c)

    st.subheader(f"Candidate Papers ({len(unique_candidates)} unique)")

    available = [c for c in unique_candidates if c.pdf_available]
    unavailable = [c for c in unique_candidates if not c.pdf_available]

    col1, col2, col3 = st.columns(3)
    col1.metric("Auto-available PDFs", len(available))
    col2.metric("Need manual upload", len(unavailable))
    col3.metric(
        "Coverage", f"{int(100 * len(available) / max(len(unique_candidates), 1))}%"
    )

    st.divider()

    # Available section
    if available:
        with st.expander(
            f"✅ Auto-available ({len(available)} papers) - Click to override with custom PDF",
            expanded=False,
        ):
            for c in available:
                col_info, col_upload = st.columns([2, 1])
                with col_info:
                    st.markdown(
                        f"**{c.paper.title[:70]}** ({c.paper.year}) "
                        f"— [{c.paper.layer.value}]"
                    )
                    if c.paper.pdf_url:
                        st.caption(f"  URL: `{c.paper.pdf_url[:80]}…`")
                with col_upload:
                    uploaded = st.file_uploader(
                        "Override PDF",
                        type=["pdf"],
                        key=f"upload_{c.paper.paper_id}",
                        label_visibility="collapsed",
                    )
                    if uploaded:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        tmp.write(uploaded.read())
                        tmp.close()
                        st.session_state.uploaded_pdfs[c.paper.paper_id] = tmp.name
                        st.success("Overridden ✓")

    # Manual upload section
    if unavailable:
        st.subheader("📤 Manual PDF Upload")
        st.caption("Upload PDFs for papers not available open-access.")
        for c in unavailable:
            col_info, col_upload = st.columns([2, 1])
            with col_info:
                st.markdown(
                    f"**{c.paper.title[:60]}** ({c.paper.year}) — {c.paper.layer.value}"
                )
                if c.paper.doi:
                    st.caption(f"DOI: {c.paper.doi}")
            with col_upload:
                uploaded = st.file_uploader(
                    "Upload PDF",
                    type=["pdf"],
                    key=f"upload_{c.paper.paper_id}",
                    label_visibility="collapsed",
                )
                if uploaded:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp.write(uploaded.read())
                    tmp.close()
                    st.session_state.uploaded_pdfs[c.paper.paper_id] = tmp.name
                    st.success("Uploaded ✓")

    st.divider()

    col_skip, col_generate = st.columns(2)
    with col_skip:
        if st.button("⏭ Skip — use auto-available only", use_container_width=True):
            st.session_state.phase = "phase_b"
            st.rerun()
    with col_generate:
        if st.button(
            "🚀 Generate Learning Document", type="primary", use_container_width=True
        ):
            st.session_state.phase = "phase_b"
            st.rerun()


# ── Phase: Phase B ─────────────────────────────────────────────────────────


def render_phase_b():
    st.title("📖 Learning Document")

    state: PipelineState = st.session_state.pipeline_state
    if not state:
        st.error("Run Phase A first.")
        return

    # Run Phase B if not done
    if not state.final_document:
        progress_area = st.empty()
        log_expander = st.expander("Terminal Logs", expanded=True)
        log_placeholder = log_expander.empty()

        def cb(msg):
            _log(msg)
            progress_area.info(f"⏳ {msg}")
            log_placeholder.code("\n".join(st.session_state.log[-20:]), language="text")

        disabled = st.session_state.disabled_gaps
        uploaded = st.session_state.uploaded_pdfs

        # Filter state to active gaps only
        state.gaps = [g for g in state.gaps if g.gap_id not in disabled]

        with st.spinner("Generating learning document…"):
            from pipeline import run_phase_b

            try:
                state = run_phase_b(
                    state=state,
                    pdf_paths=uploaded,
                    auto_fetch=True,
                    progress_callback=cb,
                )
                st.session_state.pipeline_state = state
            except RateLimitError as e:
                st.error(_rate_limit_banner(e))
                return
            except Exception as e:
                st.error(f"Phase B failed: {e}")
                return

        progress_area.success("✅ Learning document ready!")

    if state.errors:
        for err in state.errors:
            st.warning(f"Warning: {err}")

    if not state.final_document:
        st.error("Document generation failed.")
        return

    # ── Coverage report ────────────────────────────────────────────────────
    coverage = getattr(state, "_coverage_report", None)
    if coverage:
        from eval.coverage import coverage_badge

        badge, label = coverage_badge(coverage.coverage_score)
        with st.expander(
            f"{badge} Coverage Report — {label}", expanded=coverage.coverage_score < 0.9
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("✅ Covered", coverage.covered)
            c2.metric("🟡 Abstract only", coverage.abstract_only)
            c3.metric("⚠️ Uncited", coverage.uncited)
            c4.metric("❌ Missing", coverage.missing)

            if coverage.recommendations:
                for rec in coverage.recommendations:
                    icon = "✅" if "excellent" in rec.lower() else "ℹ️"
                    st.info(f"{icon} {rec}")

            # Per-gap coverage table
            rows = []
            for item in coverage.items:
                status_icon = {
                    "covered": "✅",
                    "abstract_only": "🟡",
                    "uncited": "⚠️",
                    "missing": "❌",
                }.get(item.status, "❓")
                rows.append(
                    {
                        "Status": f"{status_icon} {item.status}",
                        "Concept": item.concept,
                        "Confidence": f"{int(item.confidence * 100)}%",
                        "Citations": item.citations,
                        "Note": item.note,
                    }
                )
            if rows:
                import pandas as pd

                st.dataframe(
                    pd.DataFrame(rows), use_container_width=True, hide_index=True
                )

    # ── Stats ──────────────────────────────────────────────────────────────
    n_exps = len(state.explanations)
    n_full = sum(1 for e in state.explanations if not e.is_abstract_only)
    avg_conf = sum(e.confidence for e in state.explanations) / max(n_exps, 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Concepts explained", n_exps)
    c2.metric("Full-text sourced", f"{n_full}/{n_exps}")
    c3.metric("Avg confidence", f"{int(avg_conf * 100)}%")

    st.divider()

    # ── Document tabs ──────────────────────────────────────────────────────
    tab_doc, tab_gaps, tab_json = st.tabs(
        ["📄 Learning Document", "🔍 Gap Details", "📊 Data"]
    )

    with tab_doc:
        # Download button
        st.download_button(
            "⬇ Download Markdown",
            data=state.final_document,
            file_name=f"learning_roadmap_{state.ba_paper.paper_id[:12]}.md",
            mime="text/markdown",
        )
        st.markdown(state.final_document)

    with tab_gaps:
        for exp in state.explanations:
            status = (
                "🟡 abstract only"
                if exp.is_abstract_only
                else f"✅ {int(exp.confidence * 100)}%"
            )
            with st.expander(f"{status} — **{exp.concept}**", expanded=False):
                st.markdown(exp.explanation_text)
                if exp.source_citations:
                    st.caption("**Citations:** " + " | ".join(exp.source_citations[:8]))
                if exp.dependency_note:
                    st.info(f"Dependency: {exp.dependency_note}")

    with tab_json:
        st.json(
            {
                "ba_title": state.ba_paper.title,
                "gaps_detected": len(state.gaps),
                "ordered_gap_ids": state.ordered_gap_ids,
                "dependencies": state.dependencies,
                "explanation_stats": [
                    {
                        "concept": e.concept,
                        "confidence": e.confidence,
                        "abstract_only": e.is_abstract_only,
                        "citations": len(e.source_citations),
                    }
                    for e in state.explanations
                ],
            }
        )

    st.divider()
    if st.button("🔄 Analyse another paper"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ── Router ─────────────────────────────────────────────────────────────────

render_sidebar()

phase = st.session_state.phase
if phase == "input":
    render_input()
elif phase == "phase_a":
    render_phase_a()
elif phase == "pdf_collect":
    render_pdf_collect()
elif phase == "phase_b":
    render_phase_b()
