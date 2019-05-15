# Satellite image classification
## Image classification on sat4 and sat 6 dataset

## Requirement
`python3`

`tensorflow-gpu==1.13.1`

`numpy`

## Usage
# Training
```bash
python3 main.py --datapath ./SAT-4_and_SAT-6_datasets/sat-4-full.mat --epochs 16 --visualize_data False --mode train --output ./weights/ --batch_size 16
```
> --datapath : path to sat 4/sat6 dataset

> --epochs : Number of epochs for training the model

> --visualize_data : Option to visualize dataset

> -- mode : Two modes Train or Test

> -- output : path to output dir where models will be saved

> -- --batch_size : batchsize for model training

# Testing

```bash
python3 test.py --model_path ./weights/ --data_path ./SAT-4_and_SAT-6_datasets/sat-4-full.mat --mode predict --show_metrics True
```
> --model_path : Path to saved model : default  ./weights/

> --data_path : Path to dataset : default ./SAT-4_and_SAT-6_datasets/sat-4-full.mat

> --mode : Select mode between predict and evaluate using trained model

> --show_metrics : To visualize confusion matrix


## Dataset
Sat 4  and Sat 6 dataset contain 4 channel(r,g,b,near Infrared) Satellite images.Each image 28x28 pixels.Sat 4 and Sat 6 contain 4 and 6 label classes respectively.
Dataset can be downloaded from [here](https://www.kaggle.com/crawford/deepsat-sat4).

- Sat4 dataset
Class wise distribution of train(400000) and test images(100000) is as following:

Class labels :`barren land` , `trees` , `grassland`,`none`

![Classwise distribution train](https://github.com/Aayushktyagi/Satellite_image_classification/blob/master/Results/Train_data_sat4.png)

![Classwise distribution test](https://github.com/Aayushktyagi/Satellite_image_classification/blob/master/Results/Test_data_sat4.png)

- Sat 6 dataset
Class wise distribution of train(324000) and test(81000) is as following:

Class labesl : `building`,`barren land`,`trees`,`grassland`,`roads`,`water`

![Classwise distribution train](https://github.com/Aayushktyagi/Satellite_image_classification/blob/master/Results/Train_data_sat6.png)

![Classwise distribution test](https://github.com/Aayushktyagi/Satellite_image_classification/blob/master/Results/Test_data_sat6.png)

## Training
### Model loss

![Loss](https://github.com/Aayushktyagi/Satellite_image_classification/blob/master/Results/Sat_image_loss.png)

### Model accuracy

![Accuracy graph](https://github.com/Aayushktyagi/Satellite_image_classification/blob/master/Results/Sat_image_accuracy.png)

## Testing
- Test image and label prediction 

![Test image and prediction](https://github.com/Aayushktyagi/Satellite_image_classification/blob/master/Results/Test_image.png)

## Evaluation

### Confusion matrix on test set

![Confusion Matrix](https://github.com/Aayushktyagi/Satellite_image_classification/blob/master/Results/Confusion_matrix.png)

### classification report on test set

![Classification report](https://github.com/Aayushktyagi/Satellite_image_classification/blob/master/Results/Classification_report.png)

## Conclusion

Hence we acheived 98% test accuracy on Sat 4 dataset using convolutional neural net after 15 epochs . Dataset contains 4 channel images standard classification architecture like ResNet50 , DenseNet121 cannot be used . For a small model performance is pretty cool!! What say??

## Refrences
https://www.kaggle.com/arpandhatt/satellite-image-classification
