# base image
FROM python:3.10-slim

# set working dir
WORKDIR /app

# copy requirements
COPY ./requirements.txt /app/requirements.txt

# install libs
RUN pip install --no-cache-dir -r requirements.txt

# copy all contents
COPY . .

# expose port
EXPOSE 8000

# run
CMD ["python", "app.py"]