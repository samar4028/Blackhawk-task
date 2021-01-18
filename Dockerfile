FROM python:3.8

RUN pip install virtualenv
ENV VIRTUAL_ENV =/venv
RUN virtualenv venv -p python
ENV PATH = "$VIRTUAL_ENV/bin:$PATH"

WORKDIR /usr/src/app
COPY .  /usr/src/app




RUN pip install -r requirements.txt

CMD ["python", "predict.py"] 
