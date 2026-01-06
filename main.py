import argparse
import asyncio
import logging
from datetime import datetime
from typing import Literal
import prompts

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)
import re, os, requests

logger = logging.getLogger(__name__)

ENABLE_LOGGING=False
LOGGING_DIR="/Users/sahil/Desktop/data/metaculus/minibench_live"


class FallTemplateBot2025(ForecastBot):
    """
    This is a copy of the template bot for Fall 2025 Metaculus AI Tournament.
    This bot is what is used by Metaculus in our benchmark, but is also provided as a template for new bot makers.
    This template is given as-is, and though we have covered most test cases
    in forecasting-tools it may be worth double checking key components locally.

    Main changes since Q2:
    - An LLM now parses the final forecast output (rather than programmatic parsing)
    - Added resolution criteria and fine print explicitly to the research prompt
    - Previously in the prompt, nothing about upper/lower bound was shown when the bounds were open. Now a suggestion is made when this is the case.
    - Support for nominal bounds was added (i.e. when there are discrete questions and normal upper/lower bounds are not as intuitive)

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ones.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLMto intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions for the
    MiniBench and Seasonal AIB tournaments. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-3-5-sonnet-20241022", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/deep-research/low",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name"
    if research_strategy == "asknews/deep-research/low":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "model_name").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        # Here, we will first generate the search prompts, then search for each of them and then stitch them together as research
        async with self._concurrency_limiter:
            # Generate search query prompts
            search_query_generation_filename = f'{LOGGING_DIR}/search_query_generation/{question.id_of_question}.txt'
            if ENABLE_LOGGING and os.path.exists(search_query_generation_filename):
                search_query_generation_response = open(search_query_generation_filename).read()
            else:
                search_query_generation_prompt = prompts.get_search_query_generation_prompt(question.to_json())
                llm = GeneralLlm(model="openrouter/openai/o4-mini-high")
                search_query_generation_response = await llm.invoke(search_query_generation_prompt)
                if ENABLE_LOGGING:
                    with open(search_query_generation_filename, 'w') as f:
                        f.write(search_query_generation_response)
                
            search_queries_block = re.search(r'(?:Search queries:)(.*)', search_query_generation_response, re.DOTALL | re.IGNORECASE)
            assert search_queries_block
            queries_text = search_queries_block.group(1).strip()
            current_queries = [question.question_text]
            for query in queries_text.split('\n'):
                query = '. '.join(query.split('. ')[1:])
                if len(query) > 5:
                    current_queries.append(query)

            # Search on perplexity for these queries
            research_filename = f'{LOGGING_DIR}/perplexity_search_response/{question.id_of_question}.txt'
            if ENABLE_LOGGING and os.path.exists(research_filename):
                research = open(research_filename).read()
            else:
                current_responses = []
                PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
                for search_query in current_queries:
                    url = "https://api.perplexity.ai/chat/completions"
                    payload = {
                        "model": "sonar",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Be thorough and detailed. Be objective in your analysis, proving documented facts only. Cite all sources with names and dates."
                            },
                            {
                                "role": "user",
                                "content": search_query + " Cite all sources with names and dates, compiling a list of sources at the end. Be objective in your analysis, providing documented facts only."
                            }
                        ]
                    }
                    headers = {
                        "accept": "application/json",
                        "content-type": "application/json",
                        "authorization": f"Bearer {PERPLEXITY_API_KEY}"
                    }
                    
                    resp = requests.post(url, json=payload, headers=headers, timeout=800)
                    resp.raise_for_status()
                    data = resp.json()
                    content = data['choices'][0]['message']['content']
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                    current_responses.append(content)
                
                research = ""
                for question, answer in zip(current_queries, current_responses):
                    research += f"<Question>: {question}\n"
                    research += f"<Answer>: {answer}\n<End of Answer>\n\n"
                if ENABLE_LOGGING:
                    with open(research_filename, 'w') as f:
                        f.write(research)

            return research

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prediction_filename = f"{LOGGING_DIR}/predictions/{question.id_of_question}.txt"
        if ENABLE_LOGGING and os.path.exists(prediction_filename):
            prediction_response = open(prediction_filename).read()
        else:
            prompt = prompts.get_binary_prompt_with_research(question=question.to_json(), research=research)
            llm = GeneralLlm(model="openrouter/openai/gpt-5.2")
            prediction_response = await llm.invoke(prompt)
            if ENABLE_LOGGING:
                with open(prediction_filename, 'w') as f:
                    f.write(prediction_response)

        logger.info(f"Reasoning for URL {question.page_url}: {prediction_response}")
        binary_prediction: BinaryPrediction = await structure_output(
            prediction_response, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}"
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=prediction_response)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prediction_filename = f"{LOGGING_DIR}/predictions/{question.id_of_question}.txt"
        if ENABLE_LOGGING and os.path.exists(prediction_filename):
            prediction_response = open(prediction_filename).read()
        else:
            prompt = prompts.get_multiple_choice_prompt_with_research(question=question.to_json(), research=research)
            llm = GeneralLlm(model="openrouter/openai/gpt-5.2")
            prediction_response = await llm.invoke(prompt)
            if ENABLE_LOGGING:
                with open(prediction_filename, 'w') as f:
                    f.write(prediction_response)

        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}
            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            """
        )
        logger.info(f"Reasoning for URL {question.page_url}: {prediction_response}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=prediction_response,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}"
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=prediction_response
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        prediction_filename = f"{LOGGING_DIR}/predictions/{question.id_of_question}.txt"
        if ENABLE_LOGGING and os.path.exists(prediction_filename):
            prediction_response = open(prediction_filename).read()
        else:
            prompt = prompts.get_multiple_choice_prompt_with_research(question=question.to_json(), research=research)
            llm = GeneralLlm(model="openrouter/openai/gpt-5.2")
            prediction_response = await llm.invoke(prompt)
            if ENABLE_LOGGING:
                with open(prediction_filename, 'w') as f:
                    f.write(prediction_response)

        logger.info(f"Reasoning for URL {question.page_url}: {prediction_response}")
        percentile_list: list[Percentile] = await structure_output(
            prediction_response, list[Percentile], model=self.get_llm("parser", "llm")
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=prediction_response)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    assert run_mode in [
        "tournament",
        "metaculus_cup",
        "test_questions",
    ], "Invalid run mode"

    template_bot = FallTemplateBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        # llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
        #     "default": GeneralLlm(
        #         model="openrouter/openai/gpt-4o", # "anthropic/claude-3-5-sonnet-20241022", etc (see docs for litellm)
        #         temperature=0.3,
        #         timeout=40,
        #         allowed_tries=2,
        #     ),
        #     "summarizer": "openai/gpt-4o-mini",
        #     "researcher": "asknews/deep-research/low",
        #     "parser": "openai/gpt-4o-mini",
        # },
    )

    if run_mode == "tournament":
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        # The Metaculus cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564 or AI_2027_TOURNAMENT_ID = "ai-2027"
        # The Metaculus cup may not be initialized near the beginning of a season (i.e. January, May, September)
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            # "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            # "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            # "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029 - Discrete
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    template_bot.log_report_summary(forecast_reports)
