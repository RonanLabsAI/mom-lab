FROM vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b
ENV HF_HUB_DISABLE_XET=1 PIP_NO_CACHE_DIR=1
RUN pip install --no-deps trl==0.24.0 && pip install peft accelerate datasets sacrebleu math-verify evalplus sentencepiece liger-kernel 2>&1 | tail -3 && pip install --no-deps trl==0.24.0
WORKDIR /lab
ENTRYPOINT []
