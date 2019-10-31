# Item-Anomaly-Detection
Project to detect anomalies in item details: description, title, image and category.
This project should have the following folder structure:

```
├── Item-Anomaly-Detection (git repo)
├── data (should contain initial dataframe and unwrapped photo dataset)
├── logs (on google drive if not training by yourself)
├── models (on google drive if not training by yourself)```
```
## Steps to do to reproduce the project:
### 1.Cleaning script
```
python3 datacleaning.py
```


### 2. Scripts to train the models
```
python3 train.py --epochs 100 --model=description --dataset=description --batch-size 128 --learning-rate 0.015 --model-save-name=desc_100_015
python3 train.py --epochs 100 --model=title --dataset=title --batch-size 128 --learning-rate 0.015 --model-save-name=title_100_015
python3 train.py --epochs 50 --model=resnet_pretrained --dataset=image --batch-size 128 --learning-rate 0.002 --model-save-name=image_resnet_50_002
```
You can find __pretrained models and tensorboard logs__ here:
https://drive.google.com/drive/folders/1RBTg1NEgZ2PUN4ir1qkdf-8CY7O4sy5y?usp=sharing

### 3. Postprocessing script to obtain classification results (or you can find postprocessing results in data folder on drive)
```
python3 postprocessing.py
```

## Now everything is ready to explore and rerun 'Anomaly detection' jupyter notebook.

P.S. you might have issue when installing 'en_core_web_md'. If this happens, remove the corresponding row in requirements file and run the following command after you installed everything:
```
python3 -m spacy download en_core_web_md
```