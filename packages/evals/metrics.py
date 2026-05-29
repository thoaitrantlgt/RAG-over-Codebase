from dataclasses import asdict, dataclass
from math import log2


@dataclass(frozen=True)
class RankingMetrics:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate_ranking(
    ranked_ids: list[str],
    positive_ids: set[str],
    *,
    k_values: tuple[int, int, int] = (1, 5, 10),
) -> RankingMetrics:
    if not positive_ids:
        return RankingMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

    recalls = [recall_at_k(ranked_ids, positive_ids, k) for k in k_values]
    return RankingMetrics(
        recall_at_1=recalls[0],
        recall_at_5=recalls[1],
        recall_at_10=recalls[2],
        mrr_at_10=mrr_at_k(ranked_ids, positive_ids, 10),
        ndcg_at_10=ndcg_at_k(ranked_ids, positive_ids, 10),
    )


def recall_at_k(ranked_ids: list[str], positive_ids: set[str], k: int) -> float:
    hits = len(set(ranked_ids[:k]) & positive_ids)
    return hits / len(positive_ids)


def mrr_at_k(ranked_ids: list[str], positive_ids: set[str], k: int) -> float:
    for rank, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in positive_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], positive_ids: set[str], k: int) -> float:
    dcg = 0.0
    for rank, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in positive_ids:
            dcg += 1.0 / log2(rank + 1)

    ideal_hits = min(len(positive_ids), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def average_metrics(metrics: list[RankingMetrics]) -> RankingMetrics:
    if not metrics:
        return RankingMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

    count = len(metrics)
    return RankingMetrics(
        recall_at_1=sum(item.recall_at_1 for item in metrics) / count,
        recall_at_5=sum(item.recall_at_5 for item in metrics) / count,
        recall_at_10=sum(item.recall_at_10 for item in metrics) / count,
        mrr_at_10=sum(item.mrr_at_10 for item in metrics) / count,
        ndcg_at_10=sum(item.ndcg_at_10 for item in metrics) / count,
    )
