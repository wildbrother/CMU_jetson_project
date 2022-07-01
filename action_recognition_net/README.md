pipeline files -> 
	The files in this folder are we extracted from our docker image. 
	
There are many packages to need this codes. so if someone want to inference this codes, just use the image <wildbrother/cmu_team5_action:1.0>

README in DOCKER FILE :
	1. bash shell
		cd /opt/nvidia/deepstream/deepstream-6.0/sources/apps/sample_apps/deepstream-3d-action-recognition
	2. bash shell
		CUDA_VER=10.2 make Makefile install // if you recified cpp files.
	3. bash shell
		deepstream-3d-action-recognition -c deepstream_action_recognition_config.txt


config file -> 
	deepstream_action_recognition_config.txt
	config_infer_primary_2d_action.txt
	config_infer_primary_2d_action.txt
	config_preprocess_2d_custom.txt
	config_preprocess_3d_custom.txt


pipeline ->
	deepstream_3d_action_recognition.cpp


model -> 
	last_model.etlt
	hope2.etlt
	// there are many other weights in docker image.
	// we make this model files in TAO toolkit
	
