import json
import re

from stages.self_rag.binary_router import BinaryRouter
from utils.schemas.query import Query


class RelevanceFilterRouter(BinaryRouter):
    """Route on the per-document verdicts of RetrievalGraderFormatter and
    FILTER the documents the generator will see.

    The grader returns {"1": "yes", "2": "no", ...}. Documents graded "no" are
    dropped from `context["retrieved_documents"]` (the full set is kept in
    `context["all_documents"]`). The routing decision is "at least one
    relevant document": yes -> generate, no -> rewrite the query, with the
    same retry budget and `on_exhaust` policy as BinaryRouter (an exhausted
    query with `on_exhaust: yes` generates from everything it gathered).

    A grader output that cannot be parsed fails OPEN: every document is kept
    and the query proceeds, because a parse failure is a model formatting
    error, not evidence of irrelevance. It is counted in
    `context["grader_parse_failures"]`.

    YAML config example (same keys as BinaryRouter; retry_is_yes is fixed):
        config:
          max_retries: 2
          yes_stage_id: 5
          no_stage_id: 10
          end_stage_id: 12
          on_exhaust: yes
    """

    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)
        self._retry_is_yes = False   # a "no relevant document" verdict retries

    @staticmethod
    def _parse(text: str) -> dict | None:
        t = text.strip()
        if t.startswith("```"):
            t = "\n".join(l for l in t.split("\n") if not l.strip().startswith("```"))
        s, e = t.find("{"), t.rfind("}")
        if s == -1 or e <= s:
            return None
        try:
            obj = json.loads(t[s:e + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        out = {}
        for k, v in obj.items():
            m = re.search(r"\d+", str(k))
            if m:
                out[int(m.group())] = "yes" in str(v).lower()
        return out or None

    def _verdict(self, query: Query) -> bool:
        data = query.data
        if isinstance(data, list):
            data = data[0] if data else ""
        docs = list(query.context.get("retrieved_documents") or [])
        query.context["all_documents"] = docs
        verdicts = self._parse(str(data))
        if verdicts is None:
            query.context["grader_parse_failures"] = (
                query.context.get("grader_parse_failures", 0) + 1)
            print(f"RelevanceFilterRouter[{self.name}]: unparseable grader output, "
                  f"keeping all {len(docs)} documents: {str(data)[:80]!r}")
            return bool(docs)
        kept = [d for i, d in enumerate(docs, start=1) if verdicts.get(i, False)]
        query.context["relevant_documents"] = kept
        query.context["documents_dropped"] = (
            query.context.get("documents_dropped", 0) + (len(docs) - len(kept)))
        if kept:
            query.context["retrieved_documents"] = kept
            return True
        return False
