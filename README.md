# PYTHON [FOOTBALL PLAYER CLASSIFICATION]

# Introduction
- The gif below illustrate the dataset i will working at
- I would like to detect football player jersey number with the assumption of the problem is truth grouth bounding box

![PixellotAir-Footballonline-video-cutter com-ezgif com-resize(1)](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/4cad3d45-8b67-462b-a9c0-fd924cddc50c)


# Requirements
- pytorch
- opencv
- tensorboard

# Dataset
- My dataset include 16 classes, jersey_number from 1 to 19 without 16-17-18
- This is how my dataset-path changing working:

                 Football-dataset
                  ├── Train
                  │   ├── json_file  
                  │   ├── match_videos
                  │
                  │
                  └── Test
                      ├── json_file 
                      ├── match_videos

- Here is a orginal-image as known as a frame was cutting from one of these orginal-videos:

![Screenshot from 2024-01-12 20-45-54](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/6ad6cf7c-921b-44eb-8135-d197e4a81245)

- All the images i cropped from the original-image above:

     ![Jerseynumber6online-video-cutter com-ezgif com-gif-maker](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/b98c3b0e-8b0c-4353-8d79-a6b1dd75790d)

# Setting
- Typing and Changing from your keyboard by argparse lib
- Model structure: I was built 2 types of model were CNN model by myself and Resnet34 to compare performence
- Data augmentation: I performed dataset augmentation, to make sure that you could re-trained my model with small dataset (~450.000 cropped-images or ~4500 orginal-images). Techniques applied here includes RandomAffine, Resize and ColorJitter
- Loss: The loss function i used is CrossEntropy
- Optimizer: I used SGD optimizer

# Experiments
- I trained 2 models with NVIDIA GTX Geforce 3060 24GB GPU
- The Train loss, Validation accuracy and loss of my CNN model below:
- (I stopped training at 4 epochs)
- Training loss
![CNN model training](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/179965fd-b36e-484d-9d66-01b30945f9db)
- Validation accuracy
![CNN model val acc](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/75230d66-ddb7-4494-aa54-277f1d0b18c5)
- Validation loss
![CNN model val loss](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/65cb43bb-b399-4f7b-8562-3aa05a6c7753)

- And here is Train loss, Validation accuracy and loss of Resnet34:
- (I stopped training at 3 epochs)
- Training loss
![transfer learning training](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/02603321-fdb6-42f3-b39e-bf9953f727c3)
- Validation accuracy
![transfer learning val acc](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/3cfef884-c75b-49c8-8202-e4cdf69b6dbd)
- Validation loss
![transfer learning val losss](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/ebd2ac1e-d029-4e8a-9623-f3f438a5f002)

- My comment: Although, i was using pre-trained model is Resnet34. My CNN model still good at accuracy. It seems like my validation loss tend to overfitting cause all of training images used to training was cropped from a original-images so the quality
very low and a part of the jersey number was covering by player hand or the camera putted on the wrong place.

# Results
- test-image:
![Cristiano-Ronaldo-back-vs-Atletico-Madrid](https://github.com/TranThanhTuan2509/football-player-classification/assets/119112296/53bb1f10-9b27-4698-8a80-6434fa883300)

------------------------------------------------------Not done yet------------------------




