models=( "alexnet" "vgg16" "resnet18" "densenet121" )

for model in "${models[@]}"
do
	python eval_cnn_model.py --model ${model} --balance 1
done



