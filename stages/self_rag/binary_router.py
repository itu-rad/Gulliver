from stages.stage import Stage
from utils.schemas.query import Query


class BinaryRouter(Stage):
    """Route on a yes/no grader verdict, with a per-query retry budget.

    `retry_is_yes` says which verdict means "try again": False for the
    retrieval grader (a "no" sends the query back through the rewriter),
    True for the hallucination grader (a "yes" means the answer was
    unsupported).

    When the retry budget is exhausted the query leaves the loop. Where it
    goes is `on_exhaust`:

      end  -> `end_stage_id`, carrying the best answer produced so far in
              `query.data` (or the error marker if there is none), as the
              monolith router does.
      yes  -> `yes_stage_id` (e.g. answer anyway from whatever evidence was
              gathered)
      no   -> `no_stage_id`

    Routing to the retry branch records the answer that was rejected in
    `context["rejected_answer"]`, so a downstream rewriter can condition on
    it; under greedy decoding a retry with an unchanged prompt reproduces the
    same answer.

    YAML config example:
        config:
          max_retries: 2
          retry_is_yes: true
          yes_stage_id: 10   # rewriter
          no_stage_id: 12
          end_stage_id: 12
          on_exhaust: end
    """

    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)

        self._yes_stage_id = self.extra_config["yes_stage_id"]
        self._no_stage_id = self.extra_config["no_stage_id"]
        self._end_stage_id = self.extra_config["end_stage_id"]
        self._num_retries = self.extra_config["max_retries"]
        self._retry_is_yes = bool(self.extra_config.get("retry_is_yes", False))
        raw = self.extra_config.get("on_exhaust", "end")
        # YAML 1.1 reads a bare `yes`/`no` as a boolean; accept either spelling.
        if isinstance(raw, bool):
            raw = "yes" if raw else "no"
        self._on_exhaust = str(raw).lower()
        if self._on_exhaust not in ("end", "yes", "no"):
            raise ValueError(
                f"{self.name}: on_exhaust must be end|yes|no, got {self._on_exhaust!r}"
            )
        self._query_dict = {}

    @staticmethod
    def _verdict(query: Query) -> bool:
        """True when the grader said yes. Only the answer text is inspected --
        the inference stage strips any reasoning block before handing over."""
        data = query.data
        if isinstance(data, list):
            data = data[0] if data else ""
        return "yes" in str(data).lower()

    def _exhausted_output(self, query: Query) -> dict[int, Query]:
        query.context["retries_exhausted"] = True
        if self._on_exhaust == "yes":
            return {self._yes_stage_id: query}
        if self._on_exhaust == "no":
            return {self._no_stage_id: query}
        best = query.context.get("generated_answer")
        if isinstance(best, list):
            best = best[0] if best else None
        query.data = best if best else "Error: No more retries left"
        return {self._end_stage_id: query}

    def run(self, query: Query) -> dict[int, Query]:
        said_yes = self._verdict(query)
        retry = said_yes if self._retry_is_yes else not said_yes
        query_id = query.query_id

        if not retry:
            # Accepting verdict: for the hallucination grader the accepted
            # answer is the payload the end stage should carry.
            if self._retry_is_yes and query.context.get("generated_answer") is not None:
                query.data = query.context["generated_answer"]
            output = {self._yes_stage_id if said_yes else self._no_stage_id: query}
        else:
            if query_id not in self._query_dict:
                self._query_dict[query_id] = self._num_retries
            self._query_dict[query_id] -= 1
            if self._query_dict[query_id] < 0:
                print(f"BinaryRouter[{self.name}]: {query_id} retries exhausted "
                      f"-> {self._on_exhaust}")
                output = self._exhausted_output(query)
            else:
                if self._retry_is_yes:
                    query.context["rejected_answer"] = query.context.get("generated_answer")
                query.context["retries"] = query.context.get("retries", 0) + 1
                output = {self._yes_stage_id if said_yes else self._no_stage_id: query}
        print(f"BinaryRouter[{self.name}]: {query_id} yes={said_yes} -> {list(output)}")
        return output
