models=( "alexnet" "vgg16" "resnet18" "densenet121" )

for model in "${models[@]}"
do
	python classification_playground.py --model ${model}	
done








