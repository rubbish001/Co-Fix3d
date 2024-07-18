# Co-Fix3d & Co-Fusion & Co-Stream

welcome to replicate my experiment if you have GPUs available. I used 4 NVIDIA 4090 GPUs, and I estimate that using 8 GPUs would yield even better results.

LiDAR-Based 3D Object detection 1st on nuScenes data

Co-Fix3d mutil-mode rank 55th on the leader

Co-Fusion mutil-mode rank 25th on the leader

nuScenes test dataset 
| Method |Modality |mAP|NDS |
| ------------- | ------------- | ------------- | ------------- |
| Co-Fix3D  | L  |69.3  |72.5 |
| Co-Fix3D  | L+C  |72.3  |74.1  |
| Co-Fusion  | L+C  |74.1  |75.1  |


nuScenes val dataset 
| Method |Modality |mAP|NDS |
| ------------- | ------------- | ------------- | ------------- |
| Co-Fix3D  | L  |66.7  |71.3 |
| Co-Fix3D  | L+C  |70.6  |72.9  |
| Co-Fusion  | L+C  |73.0  |74.3  |

ToDo Multi-Frame Fusion: Co-Stream is coming soon
