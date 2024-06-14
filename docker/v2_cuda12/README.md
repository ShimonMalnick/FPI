# Docker instructions
# version 1 aims to compare different vit representations based methods, including handling vit base
1. Build the docker:
```shell
docker build -t torch:fpi_12 -f docker/v2_cuda12/Dockerfile .
docker tag torch:fpi_12 shimonmal/torch:fpi_12
docker push shimonmal/torch:fpi_12
```
2. Then run the docker:
On the A5000:
```shell
docker run -e DATASETS_ROOT=/storage/malnick/datasets -e OUTPUT_ROOT=/storage/malnick/results_test --gpus all -it -v $(pwd):/storage/malnick/ shimonmal/torch:fpi_12
```

On runai:
```shell
runai submit --name shimon-fpi-cuda12 -g 1.0 -i shimonmal/torch:fpi_12 --pvc=storage:/storage --large-shm --node-pools bengal -e DATASETS_ROOT=/storage/malnick -e OUTPUT_ROOT=/storage/malnick/results_fpi -e HF_HOME=/storage/malnick/huggingface_cache --command -- /bin/bash /storage/malnick/FPI/entry_script.sh
runai submit --name shimon-fpi-cuda12 -g 1.0 -i shimonmal/torch:fpi_12 --pvc=storage:/storage --large-shm --interactive -e DATASETS_ROOT=/storage/malnick -e OUTPUT_ROOT=/storage/malnick/results_fpi -e HF_HOME=/storage/malnick/huggingface_cache --command -- sleep infinity
runai submit --name shimon-fpi-cuda12 -g 1.0 -i shimonmal/torch:fpi_12 --pvc=storage:/storage --large-shm -e DATASETS_ROOT=/storage/malnick -e OUTPUT_ROOT=/storage/malnick/results_fpi -e HF_HOME=/storage/malnick/huggingface_cache --command -- /bin/bash /storage/malnick/FPI/entry_script.sh
```
For specific nodes, add the ```--node-pools bengal``` flag. this will run the job on the node that has H100 GPUs.
Finding out information about the different nodes can be done using:
```shell
kubectl get nodes --show-labels|grep H100
```
# useful commands
```shell
runai delete job <job-name>
runai describe job <job-name>
runai logs job <job-name>
runai bash <job-name>
```

