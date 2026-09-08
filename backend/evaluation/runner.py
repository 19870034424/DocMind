from .metrics import compute_query_metrics
from .models import EvalDataset


def _aggregate(results: list[dict]) -> dict:
    if not results:
        return {}
    keys = results[0].keys()
    return {key: sum(r[key] for r in results) / len(results) for key in keys}


class EvalRunner:
    def __init__(self):
        from services.chunker import chunker
        from storage.milvus_client import milvus_client

        self._store = milvus_client
        self._chunker = chunker

    def run(
        self,
        dataset: EvalDataset,
        k_values: list[int] | None = None,
    ) -> dict:
        if k_values is None:
            k_values = [5, 10, 20]

        vec_results: list[dict] = []
        hyb_results: list[dict] = []

        for query in dataset.queries:
            relevant = {t.strip() for t in query.relevant_chunk_texts}
            query_emb = self._chunker.embed([query.question])[0]

            vec_chunks = self._store.query(
                dataset.doc_id, query_emb, top_k=max(k_values)
            )
            vec_texts = [c["text"] for c in vec_chunks]
            vec_results.append(compute_query_metrics(vec_texts, relevant, k_values))

            hyb_chunks = self._store.hybrid_query(
                dataset.doc_id, query_emb, query.question, top_k=max(k_values)
            )
            hyb_texts = [c["text"] for c in hyb_chunks]
            hyb_results.append(compute_query_metrics(hyb_texts, relevant, k_values))

        return {
            "doc_id": dataset.doc_id,
            "document_title": dataset.document_title,
            "num_queries": len(dataset.queries),
            "k_values": k_values,
            "vector": _aggregate(vec_results),
            "hybrid": _aggregate(hyb_results),
            "per_query": {
                "vector": vec_results,
                "hybrid": hyb_results,
            },
        }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m evaluation.runner <doc_id> <dataset_path> [k1,k2,k3]")
        print("Example: python -m evaluation.runner abc123 datasets/doc_abc123.json 5,10,20")
        sys.exit(1)

    doc_id = sys.argv[1]
    dataset_path = sys.argv[2]
    k_values = (
        [int(k) for k in sys.argv[3].split(",")]
        if len(sys.argv) > 3
        else [5, 10, 20]
    )

    dataset = EvalDataset.load(dataset_path)
    if dataset.doc_id != doc_id:
        print(f"Warning: dataset doc_id '{dataset.doc_id}' != provided '{doc_id}'")

    runner = EvalRunner()
    result = runner.run(dataset, k_values)

    # Print comparison table
    separator = "=" * 65
    print(f"\n{separator}")
    print(f"  Evaluation: {dataset.document_title} ({doc_id})")
    print(f"  Queries: {len(dataset.queries)}, K values: {k_values}")
    print(separator)
    header = f"{'Metric':<20} {'Vector':>10} {'Hybrid':>10} {'Delta':>10} {'Win':>6}"
    print(header)
    print("-" * 56)
    for metric in result["vector"]:
        v = result["vector"][metric]
        h = result["hybrid"][metric]
        delta = h - v
        sign = "+" if delta >= 0 else ""
        winner = "Hybrid" if delta > 0 else ("Vector" if delta < 0 else "tie")
        print(f"{metric:<20} {v:>10.4f} {h:>10.4f} {sign}{delta:>9.4f}  {winner:>4}")
    print(separator)

    # Save detailed results
    out_path = dataset_path.replace(".json", "_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nDetailed results saved to: {out_path}")
