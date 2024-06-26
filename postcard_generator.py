import argparse
import html
import http.client
import json
import os
import random
import re
import sys
import time
import unicodedata

from hand import Hand
from drawing import alphabet

from pyaxidraw import axidraw

from card_data import CARD_DATA


# HuggingFace API wrapper ######################################################

class HFAPIException(Exception):
    pass

HF_URL = "api-inference.huggingface.co"
GEN_MODEL = "bigscience/bloom"
CLASS_MODEL = "mtheo/camembert-base-xnli"

HF_TOKEN = os.getenv("HF_TOKEN")


# Query HF API
def hf_query(payload: dict, model: str) -> dict :

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-type": "application/json",
    }

    conn = http.client.HTTPSConnection(HF_URL, timeout=10)

    try :
        conn.request(
            "POST",
            f"/models/{model}",
            json.dumps(payload),
            headers
        )
        resp = conn.getresponse()
        data = resp.read().decode()
        conn.close()

    # FIXME: bad catchall for socket.gaierror and TimeoutError
    # (and other errors ?)
    except Exception as e : 
        raise HFAPIException(e)

    if resp.status != 200 :
        raise HFAPIException(resp, data)

    parsed_json = json.loads(data)

    if "error" in parsed_json :
        raise HFAPIException(parsed_json)

    return parsed_json


# Send inference query and return generated text
def predict(prompt: str, size:int=120, seed:int=None) -> str :

    if seed is None :
        seed = random.randint(0, 999999)

    # Inference params
    parameters = {
        "max_new_tokens": size,
        "seed": seed,
        "top_k": 50,
        "top_p": 0.9,
        "temperature": 1.0,
        "do_sample": True,
        "return_full_text": False,
    }

    # Query params
    options = {
        "use_cache": True,
        "wait_for_model": True,
    }

    # Send query
    payload = {
        "inputs": prompt,
        "parameters": parameters,
        "options" : options
    }

    response = hf_query(payload, GEN_MODEL)
    return response[0]["generated_text"]


# Zero-shot classification query
def zero_shot_class(inputs: list[str], labels: list[str], multi_label=False) :
    payload = {
        "inputs": inputs,
        "parameters": { "candidate_labels": labels, "multi_label": multi_label },
        "options" :   { "use_cache": True, "wait_for_model": True },
    }

    response = hf_query(payload, CLASS_MODEL)
    return response


# Postcard generator ###########################################################

def generate_text() -> str :

    # Params
    num_cards = 5
    max_pred  = 3

    description = "Voici une collection de vieilles cartes postales de Royan de la période avant-guerre."
    separator = "<Carte postale de Royan>"

    sep_re = re.compile(r"\<[^\>]*\>?")
    seed = random.randint(0, 99999999)

    # Preprompt generation
    input_blocks = random.sample(CARD_DATA, k=num_cards)
    preprompt = (
        description 
      + "\n".join(separator + " " + block for block in input_blocks)
      + separator
    )

    # Accumulate predictions by appending to the history
    # until the begining of the next card is found
    generated = ""
    for _ in range(max_pred) :
        pred = predict(
            preprompt + generated,
            size=64,
            seed=seed,
        )

        # Don't include start of next card
        output_blocks = sep_re.split(pred, maxsplit=1)
        generated += output_blocks[0]

        # A separator was found
        if len(output_blocks) > 1 :
            break

    return generated.strip()


# Sentiment analysis ###########################################################

# Split after punctuation (including it)
# "a b ... c, d" => [ "a b ...", "c,", "d" ]
def split_sentences(text: str) -> list[str] :
    stops = [m.end() for m in re.finditer(r"[.?!;,:]+", text)]
    stops.append(len(text))

    sentences = []
    start = 0
    for s in stops :
        sentence = text[start:s].strip() #.replace("\n", " ")
        start = s

        if len(sentence) > 0 :
            sentences.append(sentence)

    return sentences


# Do zero-shot classification
def extract_sentiments(text: str) :
    # Newlines should be ignored
    sentences  = split_sentences(text.replace("\n", " "))

    labels = list(config.SENT_TAGS.keys())
    sentiments = zero_shot_class(sentences, labels)

    return sentiments


# Generate markup for TTS
def generate_ssml(sentiments) -> str :
    ssml  = "<?xml version='1.0'?>\n"
    ssml += "<speak>\n"

    for s in sentiments :
        sent = s['labels'][0]
        pre, post = config.SENT_TAGS[sent]

        inner = html.escape(s['sequence'])

        # Fix pronunciation of Royan
        # TODO: escape IPA chars ?
        inner = inner.replace("Royan", "<phoneme alphabet='ipa' ph='ʁwajɑ̃'>Royan</phoneme>")

        ssml += f"  {pre}{inner}{post}\n"

    ssml += "</speak>"
    return ssml


# Drawing ######################################################################

handwriting_model = Hand()
ad = axidraw.AxiDraw()

def normalize(text):
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if c in alphabet + ["\n"])
    return text

def format_card(text, maxlen=60):
    in_lines  = text.split("\n")
    out_lines = []

    while len(in_lines) > 0:
        l = in_lines.pop(0)

        if len(l) <= maxlen:
            out_lines.append(l)
            continue

        while len(l) > maxlen:
            # Split on space if possible
            split = maxlen
            while split >= 0 and l[split] != " ":
                split -= 1

            # Break at maxlen if not possible
            if split <= 0:
                split = maxlen
                remain = l[split:]
            else:
                remain = l[split+1:] # Don't include the space

            out_lines.append(l[:split])

            # Overflow if the next line is larger than maxlen or if the current line doesn't end with punctuation
            if len(in_lines) > 0 and (len(in_lines[0]) > maxlen or l[-1] not in ",.!?"):
                remain += " " + in_lines.pop(0)

            l = remain

        out_lines.append(l)

    return out_lines


# Entrypoint(s) ################################################################

parser = argparse.ArgumentParser(prog="Postcard Generator")
parser.add_argument("--output-dir", default="postcards")
parser.add_argument("--style", default=9, type=int)
parser.add_argument("--bias", default=10, type=float)


def main():
    args = parser.parse_args()

    base_name = time.strftime("%Y-%m-%d_%H-%M-%S") # FIXME: Avoid overriding files, check if it already existe or something, UUID, compare with last base_name...
    base_path = os.path.join(args.output_dir, base_name)

    print(f"ID: {base_name}")

    #text = generate_text()
    text = random.choice([c for c in CARD_DATA if c.count("\n") > 5])

    print("-- TEXT --")
    print(text)
    print("----------")
    with open(base_path + ".txt", "w") as f:
        f.write(text)

    text = normalize(text)
    lines = format_card(text)

    print("-- FORMAT --")
    print("\n".join(lines))
    print("------------")

    styles = [ args.style for _ in lines ]
    biases = [ args.bias  for _ in lines ]

    handwriting_model.write(
        filename=base_path+".svg",
        lines=lines,
        biases=biases,
        styles=styles,
    )

    ad.plot_setup(base_path + ".svg")
    ad.plot_run()

if __name__ == "__main__" :
    main()

