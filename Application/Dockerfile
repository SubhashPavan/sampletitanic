FROM ubuntu:16.04
LABEL maintainer="Azure App Service Container Images <appsvc-images@microsoft.com>"
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential

FROM continuumio/miniconda3
RUN conda create -n env python=3.6
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

COPY . /app
WORKDIR /app
RUN while read requirement; do conda install --yes $requirement; done < requirements.txt 

ENTRYPOINT ["python"]
EXPOSE 5000
CMD ["finished_app.py"]
