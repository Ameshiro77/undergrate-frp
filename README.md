摘要：
本文利用现有的扩散模型，搭建了一个可以用于HOI检测任务的完整数据生成管线。
该管线可以进行自动图像生成、过滤和实例标注，且相比之前的工作，该管线能够
产出许多包含罕见人-物交互实例的、可以直接进行训练的高质量图像。本文将虚
拟样本与现有样本融合得了到Syn-HICO数据集，并利用GEN-VLKT进行了训练与对
比，证明了这种方法的有效性，且该方法对于罕见的人-物交互类别的检测性能有
着较为明显的提升。

罕见类别生成结果：（左图是HICO-DET的）
![image](https://github.com/user-attachments/assets/99cd2b4c-6319-4b72-86e0-a2671c8f80c5)


虚拟样本示例：
![image](https://github.com/user-attachments/assets/49a4b9aa-e741-46d1-9241-defc7a52c9be)

本方法实验结果，从零开始实验，实线为本方法：
![image](https://github.com/user-attachments/assets/4ac764cf-01fe-4cc2-a996-9d476297031a)

本方法mAP:
![image](https://github.com/user-attachments/assets/57701239-6063-4f68-b4ff-d63554674546)
