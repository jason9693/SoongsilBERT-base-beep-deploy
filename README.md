# SoongsilBERT BEEP 

[![Run on Ainize](https://ainize.ai/images/run_on_ainize_button.svg)](https://ainize.web.app/redirect?git_repo=https://github.com/jason9693/SoongsilBERT-base-beep-deploy)

한국어 혐오성 게시글 분류모델.

### How to use

    

### Post parameter

    text: 분류할 게시글


### Output format

    {"0": result text}


## * With CLI *

### Types: logits (confidence rate)

#### Input example

    curl -X POST "#{API_URL}/predict/logits" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "text=제정신이냐?"

#### Output example

    {
        "dpstring": [],
        "result": {
            "Default": 0.052083078771829605,
            "Hate": 0.029860498383641243,
            "Offensive": 0.9180563688278198
        }
    }

### Types: class

#### Input example

    curl -X POST "#{API_URL}/predict/class" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "text=제정신이냐?"

#### Output example


    {
        "dpstring": "공격발언",
        "result": "1"
    }

