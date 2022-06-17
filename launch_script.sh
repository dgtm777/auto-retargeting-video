#!/bin/bash
ANS=""
for VARIABLE in "Fits" #"DenkaPastuh2" "HospitalTrick" "Masyanya" # "Fits" "Dogs" "Domino" "HospitalTrick" "NaskaFigurist" "NaskaFigurist2" "DenkaPastuh" "DenkaPastuh2" "Sobaki" "Richi" "Masyanya" "TheBoysScene" "Svaty1" "Svaty2" "Denka1"
do
	echo $VARIABLE
	RATIO="8:12"
	if [[ $VARIABLE == "Sobaki" || $VARIABLE == "HospitalTrick" || $VARIABLE == "Domino" || $VARIABLE == "Dogs" || $VARIABLE == "Richi" || $VARIABLE == "Denka1" ]]
	then
		RATIO="12:8"
	fi
	IN_FILENAME=$VARIABLE".mp4"
	OUT_FILENAME=$VARIABLE"_google.mp4"
	for ((i=0;i<5;i++))
	do
		./script.sh $IN_FILENAME $OUT_FILENAME $RATIO
		if [ $? -eq 0 ]
		then
			ANS=$ANS$VARIABLE" "
			break
		fi
	done
done
echo $ANS


