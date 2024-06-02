# start from a pytorch prebuilt image that suits the architecture and torch version you need
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Update the package manager and install Git
RUN apt-get -y update
RUN apt-get -y install git

RUN /bin/bash -c "pip install certifi certifi charset-normalizer contourpy cycler diffusers easydict filelock fonttools \
    fsspec idna importlib-metadata Jinja2 kiwisolver MarkupSafe matplotlib mpmath networkx numpy packaging pillow \
    psutil pyparsing python-dateutil PyYAML regex requests safetensors scipy six sympy"
RUN /bin/bash -c "pip install tokenizers torch torchvision tqdm triton typing_extensions urllib3 zipp easydict opencv-python 'transformers @ git+https://github.com/huggingface/transformers@3ec10e6c76362191b61260300fe1d6173a8dd7e1'"

# Run bash as default command
CMD ["bash"]