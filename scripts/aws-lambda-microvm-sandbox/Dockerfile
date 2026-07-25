FROM public.ecr.aws/lambda/microvms:al2023-minimal

RUN dnf install -y bash python3 && dnf clean all

WORKDIR /workspace
COPY agent.py /opt/lm-deluge/agent.py

EXPOSE 8080
CMD ["python3", "/opt/lm-deluge/agent.py"]
