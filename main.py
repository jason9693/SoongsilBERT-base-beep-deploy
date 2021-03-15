import sentencepiece
from transformers import GPT2Config, GPT2LMHeadModel
from flask import Flask, request, jsonify, render_template
import torch
from torch.nn import functional as F

from queue import Queue, Empty
from threading import Thread
import time


model_file = "every_gpt.pt"
tok_path = "kogpt2_news_wiki_ko_cased_818bfa919d.spiece"
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
                    requests["output"] = mk_everytime(requests['input'][0], requests['input'][1])
                except Exception as e:
                    requests["output"] = e


handler = Thread(target=handle_requests_by_batch).start()


def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def top_p_logits(logits, top_p=0.0, filter_value=-float('Inf')):
    """Nucleus sampling"""
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs >= top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = filter_value
    return logits


##
# GPT-2 generator.
def mk_everytime(ids, length):
    try:
        duplicate_count = 0
        duplicate_threshold = 10

        for i in range(0, length):
            input_ids = torch.tensor(ids).unsqueeze(0)
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

        result = tokenizer.decode_ids(ids)

        return result

    except Exception as e:
        print('Error occur in script generating!', e)
        return jsonify({'error': e}), 500


##
# Get post request page.
@app.route('/fairytale', methods=['POST'])
def generate():
    # GPU app can process only one request in one time.
    if requests_queue.qsize() > BATCH_SIZE:
        return jsonify({'Error': 'Too Many Requests'}), 429

    try:
        args = []

        text = request.form['text']
        length = int(request.form['length'])

        args.append(text)
        args.append(length)

    except Exception as e:
        return jsonify({'message': 'Invalid request'}), 500

    # input a request on queue
    req = {'input': args}
    requests_queue.put(req)

    # wait
    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    return jsonify(req['output'])


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