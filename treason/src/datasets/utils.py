"""
Helper function for creating datasets, LLM inference, and rejection sampling
"""

from typing import Dict, List, Optional
import re
from math_verify import parse, verify
from minerva_math_utils import get_unnormalized_answer, is_equiv, normalize_final_answer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

NUM_SAMPLES = 16


def _build_chats(batched_messages: List[List[Dict[str, str]]], system_prompt: str) -> str:
    """
    Build a chat with a system prompt, based on a list of messages
    """

    system_message = {
        "role": "system",
        "content": system_prompt,
    }

    # add the system message to all messages in the batch
    chat_messages = [[system_message] + list_messages for list_messages in batched_messages]

    return chat_messages


def _generate_batched_model_response(
    llm: LLM, system_prompt, messages_batch: List[List[Dict[str, str]]], force_think=True
):
    """
    Generate model responses for a batch of messages using vLLM.
    """
    # build chats
    chats = _build_chats(messages_batch, system_prompt)

    # apply structure if configured
    if force_think:
        think_tag_pattern = r"<think>[\s\S]+?</think>\s*<answer>[\s\S]+?</answer>$"
        structure = StructuredOutputsParams(regex=think_tag_pattern)
        sampling_params = SamplingParams(max_tokens=2048, top_p=0.95, temperature=1, structured_outputs=structure)
    else:
        sampling_params = SamplingParams(max_tokens=2048, top_p=0.95, temperature=1)

    # get model outputs
    outputs = llm.chat(chats, sampling_params, use_tqdm=True)
    return outputs


def process_results(doc: dict, attempt: str) -> dict[str, int]:

    unnormalized_answer = get_unnormalized_answer(attempt)
    answer = normalize_final_answer(unnormalized_answer)

    if is_equiv(answer, doc["answer"]):
        retval = True
    else:
        retval = False

    # math_verify
    _mvres = verify(
        gold=parse(doc["solution"]),
        target=parse(attempt),
    )
    mathval = True if _mvres else False

    res = {
        "exact_match": retval,
        "math_verify": mathval,
    }
    return res

def flexible_check(doc, response): 
    """
    Given a math problem, use a simple regex to check if the answer is correct
    """
    pattern = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
    matches = re.findall(pattern, response)
    
    if not matches:
        return False
        
    match = matches[-1]
    if isinstance(match, tuple):
        match = [m for m in match if m]
        if match:
            extracted = match[0].strip()
        else:
            return False
    else:
        extracted = match.strip()
        
    gold = str(doc["answer"])
    
    # Direct string match ignoring commas
    if extracted.replace(',', '') == gold.replace(',', ''):
        return True
        
    # Fallback to equivalence check
    return is_equiv(normalize_final_answer(extracted), gold)


def _generate_batched_model_response_prompts_only(llm: LLM, prompts: List[str], answers: Optional[List[str]] = None, samples: int = 1):
    """
    Generate model responses for a batch of messages using vLLM.
    """
    sampling_params = SamplingParams(max_tokens=2048, temperature=1.0, top_p=0.95, n=samples)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    correct_answers = [False for _ in range(len(prompts) * samples)]
    correct_answers_flexible = [False for _ in range(len(prompts) * samples)]

    if answers is not None:
        # check the answer for each prompt is correct with math_verify
        for i in range(len(prompts)):
            answer = answers[i]
            doc = {}
            doc["answer"] = answer
            # put the answer in boxed
            doc["solution"] = "\\boxed{" + answer + "}"
            for j in range(samples): 
                res = process_results(doc, outputs[i].outputs[j].text) 
                correct = res["math_verify"] or res["exact_match"]
                correct_answers[i * samples + j] = correct
                correct_answers_flexible[i * samples + j] = flexible_check(doc, outputs[i].outputs[j].text)
    else:
        # no rejection sampling
        correct_answers = [True for _ in range(len(prompts) * samples)]
        correct_answers_flexible = [True for _ in range(len(prompts) * samples)]

    return outputs, correct_answers, correct_answers_flexible

def _generate_responses_expansion_rejection_sampling(llm: LLM, prompts: List[str], samples: int = 1):
    """
    Generate model responses for a batch of messages using vLLM, with rejection sampling for prompt expansion
    """
    sampling_params = SamplingParams(max_tokens=2048, temperature=1.0, top_p=0.95, n=samples)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    correct_answers = [False for _ in range(len(prompts) * samples)]

    # rejex generated with help of Claude Opus 4.8
    rejex = re.compile(
        r"""
        \b(
            the\s+notes
        | the\s+summar(?:y|ies)
        | summarized\s+(?:solution|steps|reasoning)
        | (?:reference|provided|given)\s+solution
        | the\s+(?:solution|answer|reasoning|steps|outline|approach)\s+(?:above|given|provided)
        | (?:follow(?:ing)?|us(?:e|ing)|rely(?:ing)?\s+on)\s+the\s+
            (?:steps|reasoning|approach|outline|summary|notes)\s+
            (?:outlined|provided|given|mentioned|described|shown)\s+
            (?:in|above)
        | as\s+(?:summarized|outlined|described\s+above|shown\s+above|given\s+above|provided\s+above)
        | (?:based\s+on|according\s+to|per)\s+the\s+(?:summary|notes|solution|outline|reasoning)
        | (?:mentioned|stated|shown|noted)\s+in\s+the\s+(?:summary|notes|solution|outline)
        )\b
        """,
        re.I | re.X,
    )

    # check the answer for each prompt is correct with regex for undesired strings
    for i in range(len(prompts)):
        for j in range(samples): 
            match = rejex.search(outputs[i].outputs[j].text)
            if match:
                correct = False
            else:
                correct = True
            correct_answers[i * samples + j] = correct

    return outputs, correct_answers, None