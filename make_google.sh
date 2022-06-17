#!/bin/bash
for (( c=0; c<10; c++ ))
do
	start=`date +%s%N`
	GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/autoflip/run_autoflip --calculator_graph_config_file=mediapipe/examples/desktop/autoflip/autoflip_graph.pbtxt --input_side_packets=input_video_path=/mediapipe/$1,output_video_path=/mediapipe/$2,aspect_ratio=$3	
	end=`date +%s%N`
	if [ $? -eq 0 ]
	then
		echo "SUCCESS"
		runtime=$((end-start))
		echo $runtime
		exit 0
	fi
done
exit 1
