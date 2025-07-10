# YOLO_EWC

<img src="https://github.com/nl1xx/yolo-ewc/blob/master/EWC.png" alt="EWC" style="zoom:50%;" />

This repository implements the EWC algorithm based on the official code of **Ultralytics** to achieve continuous learning, and it is uncertain what the final effect of continuous learning will be.

## **What was modified**

1. Added `EWC.py` files to the `ultralytics/engine` directory. This class loads EWC data (Fisher Information Matrix and optimal parameters) from one or more previous tasks and computes a cumulative penalty to prevent catastrophic forgetting.
2. Added `compute_fisher.py` files and `run_training.py` files in the root directory.
3. Modified the `trainer.py` files in the `ultralytics/engine` directory to add EWC-related parts.

## **How to run**

Take the object detection task as an example.

1. **Training Task A**（**EWC is not required**）

```
yolo task=detect mode=train model=yolo11n.pt data=path/to/task_A.yaml epochs=50 name=train_A
```

2. **Calculate the EWC data for Task A**

```
python compute_fisher.py --model runs/detect/train_A/weights/best.pt --data path/to/task_A.yaml --save-path ewc_A.pt
```

3. **Training Task B** （**From this point on, EWC is needed**）

```
# ewc_data只包含上一个任务的文件
yolo task=detect mode=train model=runs/detect/train_A/weights/best.pt data=path/to/task_B.yaml epochs=50 name=train_B ewc_data=ewc_A.pt ewc_lambda=2500.0
```

4. **Calculate the EWC data for task B**

```
python compute_fisher.py --model runs/detect/train_B/weights/best.pt --data path/to/task_B.yaml --save-path ewc_B.pt
```

**And so on, you can have several different types of tasks (segmentation, detection, pose...)**

## **Future improvements**

1. **A more efficient way to integrate.**
2. **A more effective approach to continuous learning.**
3. **How effective it actually is.**

## Thanks

Thanks to the Ultralytics team, they did an excellent project.

## Reference

1. [github.com](https://github.com/ultralytics/ultralytics)
2. [arxiv.org/pdf/2109.10021](https://arxiv.org/pdf/2109.10021)

