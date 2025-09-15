docker container run --gpus device=0 --rm -ti \
--volume /home/psmeili/IS-Validation-Framework/IS_Validate:/home/psmeili/IS_Validate \
--volume /data/psmeili/Validation_Framework_Datasets/datasets:/home/psmeili/external_mount/datasets \
--volume /data/psmeili/IS_Applications/SAM2_Validate_App/:/home/psmeili/external_mount/input_application/Sample_SAM2 \
--volume /data/psmeili/Validation_Results/:/home/psmeili/external_mount/results \
--volume /home/psmeili/IS-Validation-bashscripts:/home/psmeili/validation_bashscripts \
--cpus 10 \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
--ipc host \
--name sam2v1_test \
testing:sam2v1

