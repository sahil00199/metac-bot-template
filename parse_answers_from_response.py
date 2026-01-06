from typing import Union
import re
import unicodedata
DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")
BULLET_CHARS = "•▪●‣–*-"
NUM_PATTERN = re.compile(
    r"^(?:percentile\s*)?(\d{1,3})\s*[:\-]\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*$",
    re.IGNORECASE
)
VALID_KEYS = {1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99}

def parse_answer(content, question):
    if question['question_type'] == 'binary':
        return parse_binary_probab(content)
    elif question['question_type'] in ['numeric', 'discrete']:
        return extract_percentiles_from_response(content)
    elif question['question_type'] == 'multiple_choice':
        return parse_multiple_choice_probab_distr(content, len(question['options']))




def parse_binary_probab(content):
    for line in content.split('\n')[::-1]:
        if "Probability: " in line:
            probab = line.split('Probability: ')[-1].split('%')[0].replace('*', '')
            try:
                return float(probab) * 0.01
            except:
                return -1




def parse_multiple_choice_probab_distr(content: str, num_options: int = -1) -> list[float]:
    rv = extract_option_probabilities_from_response(content=content, num_options=num_options)
    print(rv)
    rv = normalize_probabilities(rv)
    print(rv)
    return rv
    
def extract_option_probabilities_from_response(content: str, num_options: int = -1) -> list[float]:
    matches = re.findall(r"Probabilities:\s*\[([0-9.,\s]+)\]", content)
    if not matches:
        raise ValueError(f"Could not extract 'Probabilities' list from response: {content}")
    last_match = matches[-1]
    numbers = [float(n.strip()) for n in last_match.split(",") if n.strip()]
    if num_options > 0:
        if len(numbers) != num_options:
            raise ValueError(f"Expected {num_options} probabilities, got {len(numbers)}: {numbers}")
    return numbers

def normalize_probabilities(probs: list[float]) -> list[float]:
    if max(probs) > 1:
        probs = [max(min(p, 99), 1) for p in probs]
    else:
        probs = [max(min(p, 0.99), 0.01) for p in probs]
    total = sum(probs)
    normed = [p / total for p in probs]
    normed[-1] += 1.0 - sum(normed)  # minor fix for rounding
    return normed






def clean(s: str) -> str:
    # 1) Normalize compatibility forms
    s = unicodedata.normalize("NFKC", s)
    # 2) Replace every dash‐like char with ASCII hyphen
    s = DASH_RE.sub("-", s)
    # 3) Strip bullets from the start
    s = s.strip().lstrip(BULLET_CHARS)
    # 4) Remove thousands-sep commas & NBSPs
    s = s.replace(",", "").replace("\u00A0", "")
    return s.lower()
    
def extract_percentiles_from_response(content: Union[str, list], verbose: bool = False) -> dict:
    lines = content if isinstance(content, list) else content.splitlines()
    percentiles = {}
    collecting = False
    for idx, raw in enumerate(lines, 1):
        line = clean(str(raw))
        if not collecting and "distribution:" in line:
            collecting = True
            if verbose:
                print(f"🚩 Found 'Distribution:' anchor at line {idx}")
            continue
        if not collecting:
            continue
        match = NUM_PATTERN.match(line)
        if not match:
            if verbose:
                print(f"⛔ No match on line {idx}: {line}")
            continue
        key, val_text = match.groups()
        try:
            p = int(key)
            val = float(val_text)
            if p in VALID_KEYS:
                percentiles[p] = val
                if verbose:
                    print(f"✅ Matched Percentile {p}: {val}")
        except Exception as e:
            print(f"❌ Failed parsing line {idx}: {line} → {e}")
    if not percentiles:
        raise ValueError("❌ No valid percentiles extracted.")
    return percentiles