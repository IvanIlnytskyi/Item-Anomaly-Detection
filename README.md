# Item-Price-Prediction

### After requirements are installed run the following scripts to train the models
* python3 train.py --epochs 100 --model=description --dataset=description --batch-size 128 --learning-rate 0.015 --model-save-name=desc_100_015
* python3 train.py --epochs 100 --model=title --dataset=title --batch-size 128 --learning-rate 0.015 --model-save-name=title_100_015
* python3 train.py --epochs 50 --model=resnet_pretrained --dataset=image --batch-size 128 --learning-rate 0.002 --model-save-name=image_resnet_50_002