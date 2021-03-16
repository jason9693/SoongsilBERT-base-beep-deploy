FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime


RUN apt-get update && \
    apt-get install -y && \
    apt-get install -y apt-utils wget && \
    apt-get -qq -y install curl && \
    apt-get install -y tar

RUN FILEID=19t6_Cn6qPM7HEq23zbeMQdeKtqGcEz73 \
    file=kogpt2_news_wiki_ko_cased_818bfa919d.spiece \
    URL="https://drive.google.com/uc?export=download&id=$id" \
    wget --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O $file

RUN ls -l

RUN pip install --upgrade pip
RUN pip install transformers \
    flask \
    waitress \
    sentencepiece

EXPOSE 80

CMD ["python3", "main.py"]