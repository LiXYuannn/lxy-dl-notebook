# GENIMAGE: A MILLION-SCALE BENCHMARK FOR DETECTING AI-GENERATED IMAGE
- **研究背景**
   生成模型发展引发担忧，假图像传播影响社会稳定； eg：AI生成的五角大楼着火照片影响股市
  <img width="658" alt="截屏2025-04-14 11 20 42" src="https://github.com/user-attachments/assets/dfab63fd-eee5-40d0-b194-533580b9c0be" />

根据论文图表⬇️
<img width="519" alt="截屏2025-04-14 11 38 46" src="https://github.com/user-attachments/assets/9016fc40-b206-4ddb-b626-ac5ed9eea265" />

 现有假图像检测数据集存在局限，如UADFV规模小，ForgeryNet仅关注人脸、早期通用数据集依赖GAN且数据有限。
 -->GenImage数据集构建的必要性

 ## 数据集构建
 
 <img width="727" alt="截屏2025-04-14 11 42 49" src="https://github.com/user-attachments/assets/103e95ef-ea0d-424c-9ce2-4eed9e9d6d34" />

 ```包含超100万对真假图像，使用ImageNet所有真实图像，依1000个标签生成1350000张假图像```
 
 <img width="897" alt="截屏2025-04-14 11 53 16" src="https://github.com/user-attachments/assets/995ba74e-0eca-481e-9916-ce5ed7d2d646" />

用8种生成模型生成假图像，每个模型为每类生成近相同数量图像，保证数据集平衡，输入句子依ImageNet标签，部分模型输入语言有调整

<img width="576" alt="截屏2025-04-14 12 00 23" src="https://github.com/user-attachments/assets/f6033d7d-53ac-48bc-9c04-e94885ea2f3a" />

**扩散模型（diffusion model)**： Midjourney、 Wukong、 Stable Diffusio、 ADM、 GLIDE、 VQDM;

**生成对抗网络（GAN）**： BigGAN
-->其中SD V1.5最为逼真

## 数据集基准检测
 ### 假图像检测器
 **假人脸检测器（Fake Face Detector）**
专为人脸伪造检测设计，依赖人脸图像的特定特征

**代表模型**：
- F3Net：通过分析频率成分划分和真假人脸频率统计差异进行检测
- GramNet：利用全局纹理特征提升检测的鲁棒性和泛化性
  
*特点*：
- 训练数据仅为人脸图像，难以直接泛化到非人脸领域
- 设计思路可启发通用检测器的开发（eg：频率分析、纹理特征）

**通用假图像检测器（General Fake Image Detector）**
突破人脸内容的限制，检测各类假图像（如GAN或扩散模型生成）
**代表模型**：
- Spec：以频谱为输入，直接在真实图像中合成GAN伪影，无需依赖特定GAN生成的训练数据
 CNNSpot：基于ResNet-50的二分类器，通过特定的预处理、后处理和数据增强优化

*特点*：
- 现有方法在混合GAN和扩散模型生成图像的数据集上性能不足
- 急需开发针对此类混合特征的专用检测器

<img width="540" alt="截屏2025-04-14 12 04 35" src="https://github.com/user-attachments/assets/8ac2e30c-486f-4ab6-9c88-8f9ddf62eb91" />

检测器能轻松识别同一生成器生成的假图像，说明生成器会留下高度一致的痕迹（如特定频率模式、纹理特征等）。而我们需要提升检测器的泛化能力，即独立于所使用的生成器来区分图像真伪的能力。 
 →跨生成器图像分类
### 单模型 跨生成器图像分类
<img width="537" alt="截屏2025-04-14 12 05 41" src="https://github.com/user-attachments/assets/81b19caf-0e86-47bb-af4f-971ccd363e86" />

先在SD V1.4上用七种不同的方法训练的模型 然后用八种不同的检测器进行检验
//该表格可反应模型在特定训练数据下的泛化能力，根据各检测器检测准确率平均之前数据可知，**Swin-T的泛化能力最强**

### 多模型 全组合测试
<img width="507" alt="截屏2025-04-14 12 06 50" src="https://github.com/user-attachments/assets/7b6a110a-8033-494d-aba2-1a979e41b45e" />

对每个方法 都用8个生成器训练8个模型然后在8个生成器上测试并取平均值
//该测试模式反映了方法在所有可能生成器组合下的综合性能

### 退化图像处理
图像在传播过程中经常遇到**退化问题**（eg：低分辨率、压缩和噪声干扰）
<img width="294" alt="截屏2025-04-14 12 08 56" src="https://github.com/user-attachments/assets/7ab284fb-3c0b-4824-998f-b1e0a7d201e7" />

检测器应该对这些挑战具有鲁棒性

→通过评估检测器在这些退化图像上的性能，使之更准确的模拟实际条件
<img width="565" alt="截屏2025-04-14 12 09 59" src="https://github.com/user-attachments/assets/4e84de7c-0b72-4fb9-98c4-669b551836f2" />

- 作为baseline model，ResNet-50，DeiT-S和Swin-T都呈现出类似的效果 //数据十分相近
- CNNSpot对JPEG压缩和高斯模糊都具有鲁棒性 //因为CNNSpot在训练过程中使用JPEG压缩和高斯模糊作为额外的数据预处理

数据预处理即是方法论

## 数据分析
<img width="887" alt="截屏2025-04-14 18 05 14" src="https://github.com/user-attachments/assets/44998ee1-a981-4490-b0ac-735f1a8ec8c4" />

*真实图像和生成图像的频率分析对比*
- GAN伪影以规则网格的形式显示
来自扩散模型的图像比BigGAN更接近真实的图像
#### reasons：
- 在文献Adversarial Perturbations Fool Deepfake Detectors中有提到，**上采样方法**（上卷积或转置卷积）导致GAN无法正常地近似训练数据的频谱分布，所以GAN生成的图像有较多伪影
- 因为匹配较低的频率对于所生成的图像的感知质量更重要，而训练期间较少的权重被附加到较高的频率，**扩散模型不会在频谱中产生网格状伪影**，但是对于较高的频率表现出系统性的不匹配
### 为验证检测器是否能泛化到不同图像内容类别
**数据集**：
• 训练集：从GenImage的1000类中抽取子集（10、50、100），每类生成固定数量图像
• 测试集：覆盖全部1000类，每类50张生成图像，并且来自8种生成器
• 真实图像比例：每类真实图像与生成图像数量相同（平衡数据）
<img width="422" alt="截屏2025-04-14 18 12 42" src="https://github.com/user-attachments/assets/37f42f4d-91a7-40ed-950c-d045db200e5e" />

* 控制变量分析可得到，数据集标签的数量对准确度的影响程度远大于数据数量的影响程度
* 假图像检测器的泛化能力高度依赖训练数据的类别覆盖度，其中100类以上可达到较好效果
<img width="470" alt="截屏2025-04-14 18 13 44" src="https://github.com/user-attachments/assets/00b3128c-c8a4-47f4-b9d0-ad346969efd0" />

**CONCLUSION**：SD V1.4和SD V1.5与Wukong的训练产生了最佳的整体泛化性能

   GenImage范围广：不仅包含传统的人脸（face）和艺术作品（art）图像，
还涵盖更广泛的类别。
*数据来源*：
  - LFW：用于人脸识别的公开数据集，从中选取了10,000张真实人脸图像，并生成相同数量的合成人脸
  - Laion-Art：基于Laion-5B的子集，筛选出美学评分高的艺术作品，并从中选取10,000张真实艺术图像，同时生成10,000张合成艺术图像
<img width="884" alt="截屏2025-04-14 18 15 05" src="https://github.com/user-attachments/assets/5cdb1e6f-97fa-4da7-957d-5cccf6d56de4" />

### 泛化性能优异：
  - 人脸检测：99.9% 准确率（区分LFW真实人脸 vs. SDV1.4生成人脸）
  - 艺术图像检测：95.0% 准确率

**结论**：该数据集在跨内容（人脸、艺术）检测任务上表现出强泛化能力。

## 结论
- GenImage是一个专为检测生成模型生成的虚假图像而设计的大规模数据集，其规模、图像内容和生成器多样性均超越以往的数据集和基准。
- 研究提出了两项任务——跨生成器图像分类和退化图像分类，用于评估现有检测器在GenImage上的性能。
- 此外，通过对数据集的详细分析，研究揭示了GenImage如何推动开发适用于真实场景的虚假图像检测器。
