import sentencepiece
from transformers import GPT2Config, GPT2LMHeadModel
from flask import Flask, request, jsonify, render_template
import torch

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
category_map = {
    "모두의 연애": "<unused3>",
    "숭실대 에타": "<unused5>",
    "대학생 잡담방": "<unused4>"
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
                    requests["output"] = mk_everytime(requests['input'][0], requests['input'][1], requests['input'][2])
                except Exception as e:
                    requests["output"] = e


handler = Thread(target=handle_requests_by_batch).start()


##
# GPT-2 generator.
def mk_everytime(text, category, length):
    try:
        ids = tokenizer.encode_as_ids(text)

        category_id = tokenizer.piece_to_id(category_map[category])
        ids = [category_id] + ids

        input_ids = torch.tensor(ids).unsqueeze(0)

        min_length = len(input_ids.tolist()[0])

        length = length if length > 0 else 1

        length += min_length

        # story model generating
        outputs = model.generate(input_ids, pad_token_id=50256,
                                 do_sample=True,
                                 max_length=length,
                                 min_length=min_length,
                                 top_k=40,
                                 num_return_sequences=1)

        result = dict()

        for idx, sample_output in enumerate(outputs):
            result[0] = tokenizer.decode(sample_output.tolist())

        return result

    except Exception as e:
        print('Error occur in script generating!', e)
        return jsonify({'error': e}), 500


##
# Get post request page.
@app.route('/everytime', methods=['POST'])
def generate():
    # GPU app can process only one request in one time.
    if requests_queue.qsize() > BATCH_SIZE:
        return jsonify({'Error': 'Too Many Requests'}), 429

    try:
        args = []

        text = request.form['text']
        category = request.form['category']
        length = int(request.form['length'])

        args.append(text)
        args.append(category)
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