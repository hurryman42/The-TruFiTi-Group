import lmstudio as lms

from src.evaluation.base_evaluation_metric import BaseEvaluationMetric, MetricResult


class LLMAsJudge(BaseEvaluationMetric):
    def __init__(self, model: str):
        if model not in ["gpt-oss", "llama-3.1-8b", "llama-3.2-1b"]:
            raise ValueError(f"Unsupported model: {model}")
        self.model = model

        match model:
            case "gpt-oss":
                self._model_handle = "openai/gpt-oss-20b"
            case "llama-3.1-8b":
                self._model_handle = "meta-llama-3.1-8b-instruct"
            case "llama-3.2-1b":
                self._model_handle = "llama-3.2-1b-instruct"

    @property
    def name(self) -> str:
        return f"llm_as_judge-{self.model}"

    def _count_confirmations(self, response: str) -> int:
        count = 0
        for i in range(4):
            if f"{i}. Yes" in response:
                count += 1
        return count

    def compute(
        self,
        generated: list[str],
        references: list[list[str]] | None = None,
    ) -> MetricResult:
        system_prompt = """
        You are a movie critic tasked with evaluating reviews written by an AI.
        """

        message = f"""
        You are given the following review:
        "{generated}"
        
        Now answer the following questions with Yes or No:
        1. Is the reviews written correctly?
        2. Is the language of the review coherent?
        3. Is the review written in a concise way?
        4. Is the review humorous?
        """

        with lms.Client() as client:
            model = client.llm.model(self._model_handle)
            chat = lms.Chat(system_prompt)
            chat.add_user_message(message)
            result = model.respond(chat)

            return MetricResult(
                name=self.name,
                score=self._count_confirmations(str(result)),
                details={},
            )
