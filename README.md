# TeethClassifierCNN
This classifier detect if a person is showing his teeth or not

--------------------------------
QUICK STEPS
This package includes the entire code for the teeth classification Convolutional neural net , and the only guideline is in the comments inside the code.

QUICK TEST
If you don't want to run the net by configuring the environment you can see the Results folder included in this package, this is the final output of the assigned task, inside the results folder you can find two folders: showing teeth and not showing teeth, inside both of them are the images from muct-b-jpg-v1 copied to the respective folder according to the model prediction.

PERFORMANCE ON TEST SET
The net is able to classify most of the examples correctly but also is making some mistakes, most of the time this is related to the mouth detector API approach and relatively small size of training data, even with this problems I consider the model pretty accurate most of the time, here are the metrics on the test set.

|                | Predicted Negative | Predicted Positive |
|----------------|--------------------|--------------------|
| Negative Cases | TN: 259            | FP: 26             |
| Positive Cases | FN: 80             | TP: 346            |

Total samples 751

* Accuracy  0.81
* Recall  0.81
* Precision  0.93
* F1Score  0.87


TESTING THE NET
Testing the trained net requires a correctly configured Caffe Environment, please install Caffe and follow the guidelines here:
http://caffe.berkeleyvision.org/installation.html
Once you have correctly configured the environment you can test the trained Teeth net by loading the trained weights stored on the model_snapshot folder

using the following command:
python predict_feature_scaled.py

this script will classify a single image or an entire folder if you set the BULK_PREDICTION to 1 or 0, be sure to specify the paths correctly

##Teeth CNN Architecture
![alt tag](https://github.com/juanzdev/TeethClassifierCNN/blob/master/architectureTeethCNN.png)

## PIPELINE EXPLAINED
The following steps were necessary to be able to create the Teeth predictor:

1. Manual label stored on classified_data folder (752 manual labels) by using the jupyter notebook Teeth Detector.ipynb

2. Mirrored data gets generated automatically with manual labels also on classified_data folder(total 1503) x2 times data (augmented)

3. (python create_mouth_training_data.py) remove noise by extracting the mouth region only using CV2 libraries (accurate most of the time)
    3. mouth_detect_bulk from mouth_slicer.py
    3. mouth images are scaled by 50x50
    3. all data gets histogram_equalization and color to gray transforms
    3. generates all data to mouth_data folder(total 1503, some images can be incorrectly classified a mouth region CV2 libraries(114))

4. Manual cleaning of erroneous data(check if they are mouths and no eyes, also check if the cuts are correct) (delete files manually(check manually 1503 files, less than 10 min check)) , some files got corrupted , delete those , at the end it ended with 1475 files (28 files were misclassified and were deleted manually)

5. (python data_augmentation.py) Data Augmentation, I generate 8 times more data by rotating and scaling all the mouth data, all data is generated on all_data folder (total 11792)

6. Manually cut 340 images from all_data folder an put them on the validation_data folder, the rest of the images of all_data folder will be copied to training_data (training_data 11453, validation_data 340)

7. (python create_training_path_files.py) generate two textfiles containing the path for all images contained in training_data and validation_data
    7. these files are the way to create our data set in a caffe compatible format lmdb
    7. the format of the files are: path_of_file,label
    7. training_data.txt
    7. training_val_data.txt
    
8. Generate lmdb files for training set and validation set, executing
    8. remove any existing folder called train_lmdb or val_lmdb
    8. go to Teeth folder
    8. convert_imageset --gray --shuffle /devuser/Teeth/img/training_data/ training_data.txt train_lmdb
    8. convert_imageset --gray --shuffle /devuser/Teeth/img/validation_data/ training_val_data.txt val_lmdb
    
9. Compute mean of training set, this will generate a file called mean.binaryproto
    9. compute_image_mean -backend=lmdb train_lmdb mean.binaryproto
    
10. Train the Convolutional Neural Net
    10. caffe train --solver=model/solver_feature_scaled.prototxt 2>&1 | tee logteeth_ult_fe_2.log
    10. 10000 iterations trained on CPU, 5 hours, test loss 0.12 accuracy 95%
    
11. To test the net on new data
    11. python predict_feature_scaled.py with flag 1 on bulk to do the original assignment of copying all images on b folder to the RESULTS folder classified by showing teeth or not
    
