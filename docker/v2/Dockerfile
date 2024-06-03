# start from a pytorch prebuilt image that suits the architecture and torch version you need
ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

WORKDIR /storage/malnick

# this ensures that bash will run in the next RUN commands
SHELL ["/bin/bash", "-c"]

RUN conda create --name fpi python=3.10
RUN conda init bash
RUN echo "source activate fpi" > ~/.bashrc && source ~/.bashrc
RUN source activate fpi && pip install certifi

RUN apt-get update && apt-get install -y ffmpeg
RUN apt-get update && apt-get install -y libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN source activate fpi && pip install certifi charset-normalizer contourpy cycler diffusers==0.18.2 easydict filelock fonttools
RUN source activate fpi && pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN source activate fpi && pip install fsspec idna importlib-metadata Jinja2 kiwisolver MarkupSafe matplotlib mpmath networkx numpy packaging pillow
RUN source activate fpi && pip install psutil pyparsing python-dateutil PyYAML regex requests safetensors scipy six sympy tokenizers
RUN source activate fpi && pip install tqdm triton typing_extensions urllib3 zipp easydict opencv-python
RUN source activate fpi && pip install accelerate
RUN source activate fpi && pip install 'transformers @ git+https://github.com/huggingface/transformers@3ec10e6c76362191b61260300fe1d6173a8dd7e1'
