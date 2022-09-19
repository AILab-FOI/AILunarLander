FROM python:3.9.13-bullseye

RUN apt-get update && apt-get install -y python3-dev python3-pip libssl-dev locales git openssh-server cmake curl gnupg

RUN pip3 install fastapi[all] gym[all] tensorflow keras pydantic

COPY . .

EXPOSE 8007

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8007"]