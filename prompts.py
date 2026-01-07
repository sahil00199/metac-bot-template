from forecasting_tools import clean_indents
import datetime

def format_article_in_str(article):
    return f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {article['pub_date']}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

def get_binary_prompt_with_research(question, research):
    prompt_with_research = clean_indents(f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question['question_text']}

            Question background:
            {question['background_info']}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question['resolution_criteria']}

            {question['fine_print']}


            Your research assistant was asked some relevant questions to research using the internet which are documented next. Please bear in mind that these are sourced from the internet and feel free to exercise skepticism based on the sources:
            {research}

            Today is {datetime.datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            ------------------------------------------------------------------------

            Format your answer as below, it is very important to follow this format exactly, especially for the final probability, as a regex looking for 'Probability:' will be used to extract your answer.
            
            Analysis:
            {{Insert your analysis here, following the above components.}}
            
            Probability calibration
            {{Insert your calibration of your inside view prediction here.}}
            
            Checklist:
            {{Shortened, brief checklist verification here}}
            
            Probability: ZZ%
            """
        )
    return prompt_with_research

def get_binary_prompt_without_research(question):
    prompt_without_research = clean_indents(f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question['question_text']}

            Question background:
            {question['background_info']}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question['resolution_criteria']}

            {question['fine_print']}

            Today is {datetime.datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
    return prompt_without_research


def get_numeric_prompt_with_research(question, research):
    prompt_with_research = clean_indents(f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question['question_text']}

            Question background:
            {question['background_info']}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question['resolution_criteria']}

            {question['fine_print']}

            Units for answer: {question['unit_of_measure']}
            The answer is expected to be above {question['lower_bound']} and below {question['upper_bound']}. Think carefully, and reconsider your sources, if your projections are outside this range.

            Your research assistant was asked some relevant questions to research using the internet which are documented next. Please bear in mind that these are sourced from the internet and feel free to exercise skepticism based on the sources:
            {research}

            Today is {datetime.datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            
            ------------------------------------------------------------------------
            
            **Essential formatting requirements**
            (a) For large numbers, please DO NOT output commas between numbers like 1,000,000. Instead, just write 1000000. If not, this will cause a parsing error. 
            (b) You MUST prefix the final percentiles with Distribution: as a regex will be programmed to read text below 'Distribution:'. 
            
            Format your answer as below. 
            
            Analysis:
            {{Insert your analysis here, following the above components. You can segment your analysis across multiple final answer ranges if you find it useful.}}
            
            Probability calibration
            {{Insert your calibration of your inside view prediction here.}}
            
            Checklist:
            {{Shortened, brief checklist verification here}}
            
            Distribution:
            Percentile 0.1: XX
            Percentile 1: XX
            Percentile 5: XX
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            Percentile 95: XX
            Percentile 99: XX
            Percentile 99.9: XX
            """
        )
    return prompt_with_research


def get_numeric_prompt_without_research(question):
    prompt_without_research = clean_indents(f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question['question_text']}

            Question background:
            {question['background_info']}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question['resolution_criteria']}

            {question['fine_print']}

            Units for answer: {question['unit_of_measure']}
            The answer is expected to be above {question['lower_bound']} and below {question['upper_bound']}. Think carefully, and reconsider your sources, if your projections are outside this range.

            Today is {datetime.datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as a probability distribution in exactly the following format (or else automated parsing will be unable to read it):

            Distribution:
            Percentile 1: XX
            Percentile 5: XX
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            Percentile 95: XX
            Percentile 99: XX
            """
        )
    return prompt_without_research


def get_multiple_choice_prompt_with_research(question, research):
    prompt_with_research = clean_indents(f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question['question_text']}

            The options are:
            {question['options']}

            Question background:
            {question['background_info']}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question['resolution_criteria']}

            {question['fine_print']}

            Your research assistant was asked some relevant questions to research using the internet which are documented next. Please bear in mind that these are sourced from the internet and feel free to exercise skepticism based on the sources:
            {research}

            Today is {datetime.datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            
            ------------------------------------------------------------------------
            
            Format your answer as below and be sure to follow the formatting requirements else automated regex extraction will fail.
            
            Analysis:
            {{Insert your analysis here, following the above components. You can segment your analysis across multiple categories of options if you find it useful.}}
            
            Probability calibration
            {{Insert your calibration of your inside view prediction here.}}
            
            Checklist:
            {{Shortened, brief checklist verification here}}
            
            Probabilities: [Probability_A, Probability_B, ..., Probability_N]
            """
        )
    return prompt_with_research

def get_multiple_choice_prompt_without_research(question):
    prompt_without_research = clean_indents(f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question['question_text']}

            The options are:
            {question['options']}

            Question background:
            {question['background_info']}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question['resolution_criteria']}

            {question['fine_print']}

            Today is {datetime.datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as a probability distribution in exactly the following format (or else automated parsing will be unable to read it):

           Probabilities: [Probability_A, Probability_B, ..., Probability_N]
           """
        )
    return prompt_without_research
    
def get_prompt_with_research(question, research):
    if question['question_type'] == 'binary': return get_binary_prompt_with_research(question, research)
    elif question['question_type'] in ['numeric', 'discrete']: return get_numeric_prompt_with_research(question, research)
    elif question['question_type'] == 'multiple_choice': return get_multiple_choice_prompt_with_research(question, research)
    else:
        raise ValueError(f"Unsupported question_type {question['question_type']}")

def get_prompt_without_research(question):
    if question['question_type'] == 'binary': return get_binary_prompt_without_research(question)
    elif question['question_type'] in ['numeric', 'discrete']: return get_numeric_prompt_without_research(question)
    elif question['question_type'] == 'multiple_choice': return get_multiple_choice_prompt_without_research(question)
    else:
        raise ValueError(f"Unsupported question_type {question['question_type']}")




def get_search_query_generation_prompt(question):
    prompt = clean_indents(f"""
        You are currently doing research for historical information on the below forecasting question.
        
        The forecasting question is:
        {question['question_text']}
        
        Question background:
        {question['background_info']}
        
        This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
        {question['resolution_criteria']}
        
        Additional fine-print:
        {question['fine_print']}
        
        Today is {datetime.datetime.now().strftime("%Y-%m-%d")}.
        
        Your task is to analyze the forecasting question and write a series of search queries that will be used by your assistant to find relevant historical context and currently relevant developments or new articles using Perplexity.
        Your query will be processed by a reasoning model equipped with capable web crawlers and designed to generate lengthy, detailed responses. As such, you may use a longer query with detailed instructions. It is possible to ask multiple questions. 
        Nonetheless, you are advised to keep your query to at most four sentences.
        
        You should format your answer exactly as below. Do not wrap your query in quotes. Be sure to include at least 2 questions but no more than 5 questions in decreasing order of relevance.
        
        Analysis:
        {{Your initial impression/analysis of the forecasting question followed by reasoning about the most relevant historical context needed to generate an outside view.}}
        
        Search queries:
        1. [Query details]
        2. [Query details]
        3. [Query details]
        Any more queries
        
        """)
    return prompt


    

    