FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/oracle-jdk8-installer;


RUN apt-get update && apt-get install -y python3.8 python3-pip

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64
RUN export JAVA_HOME
ENV PYTHONPATH /usr/bin/python3.8
RUN export PYTHONPATH
WORKDIR /

RUN alias python3=/usr/bin/python3.8

COPY requirements.txt requirements.txt
COPY hp544_prediction.py hp544_prediction.py
RUN python3.8 -m pip install --upgrade pip
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y curl vim wget

RUN wget --no-verbose -O apache-spark.tgz "https://dlcdn.apache.org/spark/spark-3.2.4/spark-3.2.4-bin-hadoop3.2.tgz" \
&& mkdir -p /opt/spark \
&& tar -xf apache-spark.tgz -C /opt/spark --strip-components=1 \
&& rm apache-spark.tgz

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-arm64
RUN export JAVA_HOME

ENV SPARK_HOME="/opt/spark"
ENV PATH="${SPARK_HOME}/bin/:${PATH}"
RUN export PATH="$JAVA_HOME:$SPARK_HOME:$PATH"

RUN export SPARK_LOCAL_IP="127.0.0.1"
RUN export PATH=$JAVA_HOME/bin:$SPARK_HOME:$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
RUN export PYSPARK_PYTHON=/usr/bin/python3.8

RUN export PYSPARK_SUBMIT_ARGS="--master local[2] pyspark-shell"


CMD ["python3.8", "hp544_prediction.py"]