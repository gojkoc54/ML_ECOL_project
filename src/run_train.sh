models=( "alexnet" "vgg16" "resnet18" "densenet121" )

for model in "${models[@]}"
do
	python train_cnn_model.py --model ${model} --epochs 10 --lr 0.0001
done


