# Docker instructions
# ver 1 aims to compare different vit representations based methods, including handling vit base
1. Build the docker:
```shell
docker build -t torch:fpi -f docker/v2/Dockerfile .
docker tag torch:fpi shimonmal/torch:fpi
docker push shimonmal/torch:fpi
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all -it -v $(pwd):/FPI shimonmal/torch:fpi
```

On runai:
```shell
runai submit --name shimon-fpi -g 1.0 -i shimonmal/torch:fpi --pvc=storage:/storage --large-shm -e COCO_ROOT=/storage/malnick/mscoco17 -e CHEST_XRAY_ROOT=/storage/malnick/chest_xray
runai submit --name shimon-fpi -g 1.0 -i shimonmal/torch:fpi --pvc=storage:/storage --large-shm --interactive -e COCO_ROOT=/storage/malnick/mscoco17 -e CHEST_XRAY_ROOT=/storage/malnick/chest_xray --command -- sleep 300
```

# useful commands
```shell
runai delete job <job-name>
runai describe job <job-name>
runai logs job <job-name>
runai bash <job-name>
```


