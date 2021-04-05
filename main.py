import sentencepiece
from transformers import GPT2Config, GPT2LMHeadModel
from flask import Flask, request, render_template
import torch
from torch.nn import functional as F
import traceback

import os
from queue import Queue, Empty
from threading import Thread
import time

model_file = "./every_gpt.pt"
tok_path = "./kogpt2_news_wiki_ko_cased_818bfa919d.spiece"
kogpt2_config = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "vocab_size": 50000,
    "activation_function": "gelu"
}
category_map = {
    "모두의 연애": "<unused3>",
    "숭실대 에타": "<unused5>",
    "대학생 잡담방": "<unused4>"
}
os.system('ls')
app = Flask(__name__)

# Model & Tokenizer loading
tokenizer = sentencepiece.SentencePieceProcessor()
tokenizer.load(tok_path)

model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path=None,
                                        config=GPT2Config.from_dict(kogpt2_config),
                                        state_dict=torch.load(model_file))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

requests_queue = Queue()    # request queue.
BATCH_SIZE = 1              # max request size.
CHECK_INTERVAL = 0.1


##
# Request handler.
# GPU app can process only one request in one time.
def handle_requests_by_batch():
    while True:
        request_batch = []

        while not (len(request_batch) >= BATCH_SIZE):
            try:
                request_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for requests in request_batch:
                try:
                    requests["output"] = mk_everytime(requests['input'][0], requests['input'][1], requests['input'][2])
                except Exception as e:
                    requests["output"] = e


handler = Thread(target=handle_requests_by_batch).start()


##
# top_k_logits
def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


##
# GPT-2 natural generator
def mk_natural_everytime(ids):
    duplicate_count = 0
    duplicate_threshold = 10

    for i in range(0, 512):
        input_ids = torch.tensor(ids).unsqueeze(0)
        input_ids = input_ids.to(device)
        pred = model(input_ids)[0]
        logits = pred[:, -1, :]
        # logits = top_p_logits(logits, 0.8)
        logits = top_k_logits(logits, 10)
        log_probs = F.softmax(logits, dim=-1)
        prev = torch.multinomial(log_probs, num_samples=1)
        gen = prev[0].tolist()
        if gen[0] == tokenizer.eos_id():
            break
        duplicate_count = duplicate_count + 1 if ids[-1] == gen[0] else 0
        if duplicate_count > duplicate_threshold:
            break
        ids += gen

    result = tokenizer.decode_ids(ids[1:]).replace('<unused2>', '\n').replace('<unused0>', 'https://...')

    return result


##
# GPT-2 generator.
def mk_everytime(text, category, length):
    try:
        length = length if length > -1 else 0

        ids = tokenizer.encode_as_ids(text)
        category_id = tokenizer.piece_to_id(category_map[category])
        ids = [category_id] + ids

        result = dict()

        if length == 0:
            result[0] = mk_natural_everytime(ids)
        else:
            input_ids = torch.tensor(ids).unsqueeze(0)
            input_ids = input_ids.to(device)

            min_length = len(input_ids.tolist()[0])

            length += min_length

            # model generating
            outputs = model.generate(input_ids, pad_token_id=50256,
                                     do_sample=True,
                                     max_length=length,
                                     min_length=min_length,
                                     top_k=40,
                                     num_return_sequences=1)

            for idx, sample_output in enumerate(outputs):
                result[0] = tokenizer.decode(sample_output[1:].tolist()).replace('<unused2>', '\n').replace('<unused0>', 'https://...')

        return result, 200

    except Exception as e:
        traceback.print_exc()
        return {'error': e}, 500


##
# Get post request page.
@app.route('/everytime', methods=['POST'])
def generate():
    # GPU app can process only one request in one time.
    if requests_queue.qsize() > BATCH_SIZE:
        return {'Error': 'Too Many Requests'}, 429

    try:
        args = []

        text = request.form['text']
        category = request.form['category']
        length = int(request.form['length'])

        args.append(text)
        args.append(category)
        args.append(length)

    except Exception as e:
        return {'message': 'Invalid request'}, 500

    # input a request on queue
    req = {'input': args}
    requests_queue.put(req)

    # wait
    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    return req['output']


##
# Queue deadlock error debug page.
@app.route('/queue_clear')
def queue_clear():
    while not requests_queue.empty():
        requests_queue.get()

    return "Clear", 200


##
# Sever health checking page.
@app.route('/healthz', methods=["GET"])
def health_check():
    return "Health", 200


##
# Main page.
@app.route('/')
def main():
    return render_template('main.html'), 200


if __name__ == '__main__':
    from waitress import serve
    serve(app, port=80, host='0.0.0.0')