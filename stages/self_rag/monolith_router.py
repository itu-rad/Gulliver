import json

from stages.stage import Stage
from utils.schemas.query import Query


class MonolithRouter(Stage):
    """Routes the monolith LLM's JSON output based on grading results.

    Parses the JSON output from the monolith LLM and decides:
    - If relevance_grade == "yes" AND hallucination_check == "no" → accept (end)
    - Otherwise → retry via query rewriting (if retries remaining)
    - If retries exhausted → end

    Follows the same retry-tracking pattern as BinaryRouter.

    YAML config example:
        config:
          accept_stage_id: 5
          retry_stage_id: 6
          end_stage_id: 5
          max_retries: 2
    """

    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)

        self._accept_stage_id = self.extra_config["accept_stage_id"]
        self._retry_stage_id = self.extra_config["retry_stage_id"]
        self._end_stage_id = self.extra_config["end_stage_id"]
        self._max_retries = self.extra_config.get("max_retries", 2)
        self._query_retries = {}

    def _parse_json_output(self, raw_output: str) -> dict:
        """Attempt to parse JSON from the LLM output.

        Handles common LLM quirks: extra text around JSON, markdown fences, etc.
        """
        text = raw_output.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        print(f"MonolithRouter: failed to parse JSON from: {text[:200]}")
        return {}

    def run(self, query: Query) -> dict[int, Query]:
        """Parse the monolith JSON output and route accordingly."""

        raw_output = query.data
        if isinstance(raw_output, list):
            raw_output = raw_output[0]

        parsed = self._parse_json_output(str(raw_output))

        relevance = parsed.get("relevance_grade", "no").lower().strip()
        hallucination = parsed.get("hallucination_check", "yes").lower().strip()
        answer = parsed.get("answer", "")

        query_id = query.query_id

        # `best_answer` keeps the last non-empty answer across retries, so an
        # exhausted query leaves with it, as in BinaryRouter.
        query.context["generated_answer"] = answer
        query.context["monolith_result"] = parsed
        if answer:
            query.context["best_answer"] = answer

        is_acceptable = (
            (relevance == "yes") and (hallucination == "no") and len(answer) > 0
        )

        if is_acceptable:
            print(
                f"MonolithRouter: query {query_id} accepted (relevance={relevance}, hallucination={hallucination})"
            )
            query.data = answer
            return {self._accept_stage_id: query}
        else:
            if query_id not in self._query_retries:
                self._query_retries[query_id] = self._max_retries

            self._query_retries[query_id] -= 1

            if self._query_retries[query_id] < 0:
                print(f"MonolithRouter: query {query_id} exhausted retries, ending")
                query.context["retries_exhausted"] = True
                best = query.context.get("best_answer") or answer
                query.data = (
                    best if best else "Error: no satisfactory answer after retries"
                )
                return {self._end_stage_id: query}
            else:
                if answer:
                    query.context["rejected_answer"] = answer
                query.context["retries"] = query.context.get("retries", 0) + 1
                print(
                    f"MonolithRouter: query {query_id} retry "
                    f"(remaining={self._query_retries[query_id]}, "
                    f"relevance={relevance}, hallucination={hallucination})"
                )
                return {self._retry_stage_id: query}
