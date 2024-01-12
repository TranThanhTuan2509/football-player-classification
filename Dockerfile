FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

RUN pip install opencv-python scikit-learn numpy matplotlib tensorboard torchinfo
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY training.py /workspace/CNNtraining.py
COPY models.py /workspace/model.py
COPY dataset.py /workspace/dataset.py
COPY transfer_learning.py /workspace/transfer_learning.py

CMD ["python3", "transfer_learning.py"]
