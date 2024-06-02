# start from a pytorch prebuilt image that suits the architecture and torch version you need
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /storage

RUN apt-get update && apt-get install -y ffmpeg
RUN apt-get update && apt-get install -y libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN /bin/bash -c "pip install certifi certifi charset-normalizer contourpy cycler diffusers==0.18.2 easydict filelock fonttools \
    fsspec idna importlib-metadata Jinja2 kiwisolver MarkupSafe matplotlib mpmath networkx numpy packaging pillow \
    psutil pyparsing python-dateutil PyYAML regex requests safetensors scipy six sympy tokenizers torch torchvision tqdm triton typing_extensions urllib3 zipp easydict opencv-python accelerate 'transformers @ git+https://github.com/huggingface/transformers@3ec10e6c76362191b61260300fe1d6173a8dd7e1'"

# Run bash as default command
CMD ["bash"]