FROM python:3.11.5
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
#RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install opencv-python-headless
EXPOSE 5000
CMD [ "flask", "run", "--host=0.0.0.0"]


