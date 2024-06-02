FROM shimonmal/torch:base
WORKDIR /storage/malnick


RUN echo "source activate base" > ~/.bashrc
RUN /bin/bash -c "pip install certifi certifi charset-normalizer contourpy cycler diffusers easydict filelock fonttools"
RUN /bin/bash -c "pip install fsspec idna importlib-metadata Jinja2 kiwisolver MarkupSafe matplotlib mpmath networkx numpy"
RUN /bin/bash -c "pip install packaging pillow psutil pyparsing python-dateutil PyYAML regex requests safetensors scipy six sympy"
RUN /bin/bash -c "pip install tokenizers torch torchvision tqdm triton typing_extensions urllib3 zipp easydict opencv-python 'transformers @ git+https://github.com/huggingface/transformers@3ec10e6c76362191b61260300fe1d6173a8dd7e1'"
# Run bash as default command
CMD ["bash"]
