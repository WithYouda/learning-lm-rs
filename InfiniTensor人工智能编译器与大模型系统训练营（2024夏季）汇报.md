---
created: 2024-09-29T09:59
updated: 2024-10-08T09:49
---
# InfiniTensor: 人工智能编译器与大模型系统训练营（2024夏季）——汇报
@Author: 肖道平/ WithYouda
@Email: 3473007149@qq.com

项目地址：[WithYouda/learning-lm-rs ](https://github.com/WithYouda/learning-lm-rs)

## 工作内容简介
### 推理系统算子实现
由于是第一次接触相关算子，最开始连公式都快看不懂 ：） 
最后在推导一遍又一遍后，才初步完成了专业阶段的代码

### 模型生成服务
#### stroy 实现
在这里，十分感谢候磊同学提供的资料，最开始我对self-attention 的部分十分头痛，对着候磊同学的资料推导了一遍又一遍之后，才发现我的部分算子实现错了😅
改正算子之后，self-attention的结果终于正常了，路途艰辛呀 ：）
stroy 生成没遇到什么困难，基本按照main函数的提示，完成generate 函数之后，story生成就正常了
如图所示：
![6aca62dd3f066cfdb60df0ec526ceae.png](https://obsidian-for-me-image.oss-cn-chengdu.aliyuncs.com/6aca62dd3f066cfdb60df0ec526ceae.png)

#### chat 实现
chat 实现没什么难度，完善好model.rs文件中的chat代码就可以了，改进了输出的处理
效果如图所示
![image.png](https://obsidian-for-me-image.oss-cn-chengdu.aliyuncs.com/20240929172120.png)


### FP16 半精度推理的实现
感谢项目群@躺平同学的思路提醒，使用num-traits加上half库，对传入参数用Float抽象，就可使实现同时兼容bf16, f16 及 f32
在此过程中，遇到了无数的困难，很多代码都需要改动，但是rust学艺不精，一直在与编译器斗智斗勇😂
最后找到了精度损失的问题所在：
rope函数的freq计算过程中
$freq = pos / theta ^ {i * 2 \over d}$
最开始没有将 i 与 d 转为T泛型类型进行计算，而是直接将二者的计算结果转为了T类型
代码在f16分支上
最后bf16 stroy运行结果，如下图所示：
![image.png](https://obsidian-for-me-image.oss-cn-chengdu.aliyuncs.com/20241008094457.png)
限于时间关系及能力，暂时没有测试f16的支持情况



## 总结与反思
在这次训练营中，我学习到了llama相关的知识，这也是我第一次接触到人工智能领域，我作为一名大学生，非常荣幸并且十分愿意投身到人工智能发展的浪潮之中，这次训练营只是起点，对我来说，探索人工智能世界脚步才刚刚迈出。
虽然我在这次的训练营中表现不佳，没有完成自己预订的目标，但是通过项目给我的历练没有白费，这种经验也是我最需要的。
最后，非常感谢各位帮助过我的同学和老师，谢谢你们！


