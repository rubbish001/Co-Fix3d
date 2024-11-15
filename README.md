# BEVFix& Co-Fix3d & Co-Fusion & Co-Stream

The source code will be made available upon acceptance of the paper.

welcome to replicate my experiment if you have GPUs available. I used 4 NVIDIA 4090 GPUs, and I estimate that using 8 GPUs would yield even better results.

LiDAR-Based 3D Object detection 1st on nuScenes data
![image](https://github.com/user-attachments/assets/de987d56-41c1-416b-8dd6-de78e8412a6e)

Co-Fix3d mutil-mode rank 55th on the leader
![image](https://github.com/user-attachments/assets/aa84d720-59d6-41c5-9a26-c4f400f289c5)

Co-Fusion mutil-mode rank 25th on the leader
![1721280421256](https://github.com/user-attachments/assets/3813c88b-0c46-4583-986b-51d49a9b0733)

nuScenes test dataset 
| Method |Modality |mAP|NDS | Ref
| ------------- | ------------- | ------------- | ------------- |------------- |
| BEVFix  | L  |68.0  |71.9 |
| Co-Fix3D  | L  |69.4  |73.5 |[result](https://evalai.s3.amazonaws.com/media/submission_files/submission_481792/75e2db36-512e-49e3-9499-c0c6fd0f613f.json)|
| Co-Fix3D  | L+C  |72.3  |74.7  |[result](https://evalai.s3.amazonaws.com/media/submission_files/submission_482475/b8eb0314-2419-4533-b2d7-b006c35faf3e.json)||
| Co-Fusion  | L+C  |74.1  |75.3  ||


nuScenes val dataset 
| Method |Modality |mAP|NDS |log |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| BEVFix  | L  |65.5  |70.0 ||
| Co-Fix3D  | L  |67.3  |72.0 |[log](https://drive.google.com/file/d/1awhWDqwUsKc08f3_4F874YV1brpC9S3k/view?usp=drive_link)|
| Co-Fix3D  | L+C  |70.8  |73.6  ||
| Co-Fusion  | L+C  |73.5  |74.9  |[log](https://drive.google.com/file/d/1gdrjTm1l7gUpTvee13XJtB-YXp52makT/view?usp=drive_link)|

ToDo Multi-Frame Fusion: Co-Stream is coming soon
