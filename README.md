# [PYTHON] FOOTBALL PLAYER CLASSIFICATION

# Introduction
- The gif below illustrates the dataset I will be working with
- I would like to detect the football-player-jersey number with the assumption of the problem is ground truth bounding box

- My second model explanation following to:

- If you wanna using the second model you must change the path and call lib by yourself.
- My result model has 3 outputs (digit, digits, unknown) - (11 numbers from 0 to 9) - (color 0 - 1 black - white)
- if the first output is 0 and the second output from 0 to 9 it means the result is 0 to 9, but if the first output is 1 and the second output from 0 to 9 it means the number jersey is 10 to 19.

![PixellotAir-Footballonline-video-cutter com-ezgif com-resize(1)](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/4cad3d45-8b67-462b-a9c0-fd924cddc50c)

# Requirements
- pytorch
- opencv
- tensorboard

# Dataset
- My dataset includes 16 classes, jersey_number from 1 to 19 without 16-17-18
- This is how my dataset-path changing works:

                 Football-dataset
                  ├── Train
                  │   ├── json_file (annotation file)  
                  │   ├── match_videos
                  │
                  │
                  └── Test
                      ├── json_file (annotation file) 
                      ├── match_videos

- Here is an orginal-image as known as a frame that was cutting from one of these original-videos:

![Screenshot from 2024-01-12 20-45-54](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/6ad6cf7c-921b-44eb-8135-d197e4a81245)

- All the images I cropped from the original image above:

     ![Jerseynumber6online-video-cutter com-ezgif com-gif-maker](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/b98c3b0e-8b0c-4353-8d79-a6b1dd75790d)

# Setting
- Typing and Changing params from your keyboard by argparse lib
- Model structure: I built 2 types of models were CNN model by myself and Resnet34 to compare performance
- Data augmentation: I performed dataset augmentation, to make sure that I could increase my model performance (~450.000 cropped images or ~4500 original images).Despite, a lot of training-images but the rate of quality is very low. Techniques applied here include RandomAffine, Resize, and ColorJitter
- Loss: The loss function I used is CrossEntropy
- Optimizer: I used SGD optimizer

# Experiments
- I trained 2 models with NVIDIA GTX Geforce 3060 24GB GPU
- The Train loss, Validation accuracy, and loss of my CNN model are below:
- (I stopped training at 4 epochs)
- Training loss
![CNN model training](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/179965fd-b36e-484d-9d66-01b30945f9db)
- Validation accuracy
![CNN model val acc](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/75230d66-ddb7-4494-aa54-277f1d0b18c5)
- Validation loss
![CNN model val loss](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/65cb43bb-b399-4f7b-8562-3aa05a6c7753)

- Here is Train loss, Validation accuracy, and loss of Resnet34:
- (I stopped training at 3 epochs)
- Training loss
![transfer learning training](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/02603321-fdb6-42f3-b39e-bf9953f727c3)
- Validation accuracy
![transfer learning val acc](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/3cfef884-c75b-49c8-8202-e4cdf69b6dbd)
- Validation loss
![transfer learning val losss](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/ebd2ac1e-d029-4e8a-9623-f3f438a5f002)

- My comment: Although I was using a pre-trained model, Resnet34, my CNN model still performs well in terms of accuracy. It appears that my validation loss tends to indicate overfitting, as all the training images used were cropped from original images, resulting in low quality. Additionally, a portion of the jersey number was covered by the player's hand or the camera was placed in an incorrect position.

# Result
- test-image:
![Cristiano-Ronaldo-back-vs-Atletico-Madrid](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/53bb1f10-9b27-4698-8a80-6434fa883300)

- result:
![Screenshot from 2024-03-07 13-02-35](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/f91cd8bf-0ac4-4194-abb0-bbd20dae276a)
- 0 stands for digit, 7 stands for the number jersey is 7, and 1 is a white color
- test-image:

![croppedimage](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/b92f3be8-a2e1-4777-a153-5c1654329bb7)

- result:
![Screenshot from 2024-03-07 15-54-46](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/5374af9b-987c-4fdf-93dc-ce863cc21d3e)
- 0 stands for digit, 5 stands for the number jersey 5 but the actual number is 3, and 1 is a white color
- My comment: My test image and the result have a big difference. Like I said above the accuracy of my CNN model was very low so the wrong result was unsurprised.

# In Conclusion

- This project has only purpose is to summarize all knowledge that i learned about deep learning, especially about image classification. I appreciate who take your time for my project.


