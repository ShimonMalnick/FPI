# Docker instructions
1. Build the docker:
```shell
docker build -t torch:base -f docker/base/Dockerfile .
docker tag torch:base shimonmal/torch:base
docker push shimonmal/torch:base
```
2. Then run the docker: <br>(run from the directory above the folder "FPI") <br> On the A5000:
```shell
docker run --gpus all -it -v $(pwd):/FPI shimonmal/torch:base
```
On runai:
```shell
runai submit --name shimon-fpi -g 1.0 -i shimonmal/torch:base --pvc=storage:/storage --large-shm
```
next line is an interactive job
```shell
runai submit --name shimon-fpi -g 1.0 -i shimonmal/torch:base --pvc=storage:/storage --large-shm --interactive --command -- sleep 300
runai bash shimon-fpi
```

# useful commands
```shell
runai delete job <job-name>
runai describe job <job-name>
runai logs job <job-name>
```

