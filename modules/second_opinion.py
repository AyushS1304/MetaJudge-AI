"""
Compatibility wrapper around the shared deep verifier.

Historically the CLI and Streamlit app used different fallback implementations.
Keeping this module means existing imports continue to work while both paths now
share the same logic in ``modules.deep_verifier``.
"""

from __future__ import annotations

from modules.deep_verifier import deep_verify


def get_second_opinion(
    fact: str,
    context: str,
    original_evidence: str,
    known_papers: dict,
    verbose: bool = True,
) -> dict | None:
    return deep_verify(
        fact,
        context,
        original_evidence,
        known_papers,
        verbose=verbose,
    )


if __name__ == "__main__":
    from modules.query_generator import generate_skeptical_queries
    from modules.retriever import KNOWN_PAPERS, format_evidence_block, retrieve_evidence

    fact = "The paper demonstrated results with a rank of 8, achieving 91.3% accuracy on MNLI."
    context = "LoRA was proposed by Hu et al. from Microsoft in 2022."

    queries = generate_skeptical_queries(fact)
    evidence = retrieve_evidence(queries, fact=fact, context=context)
    evidence_block = format_evidence_block(evidence)

    result = get_second_opinion(fact, context, evidence_block, KNOWN_PAPERS)
    if result:
        print(f"\nVerdict:  {result.get('verdict')}")
        print(f"Reasoning: {result.get('reasoning')}")
        print(f"PDF used:  {result.get('pdf_used')}")
    else:
        print("No second opinion available")
