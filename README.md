# <p align="center">SeIF: Semantic-constrained Deep Implicit Function for Single-image 3D Head Reconstruction</p>
#### <p align="center">Leyuan Liu, Xu Liu, Jianchi Sun, Changxin Gao, Jingying Chen</p>
***
  Various applications require 3D avatars that are realistic, artifact-free, and animatable. However, traditional 3D morphable models (3DMMs) produce animatable 3D heads but fail to capture accurate geometries and details, while existing deep implicit functions have been shown to achieve realistic reconstructions but suffer from artifacts and struggle to yield 3D heads that are easy to animate. To reconstruct [**high-fidelity**](#RESULTS), [**artifact-less**](#RESULTS), and [**animatable**](#Animation) 3D heads from single images, we leverage dense semantics to bridge the best properties of 3DMMs and deep implicit functions and propose SeIF---a semantic-constrained deep implicit function. First, SeIF derives dense semantics from a standard 3DMM (e.g., FLAME) and samples a semantic code for each query point in the query space to provide a soft constraint to the deep implicit function. The reconstruction results show that this semantic constraint does not weaken the powerful representation ability of the deep implicit function while significantly suppressing artifacts. Second, SeIF predicts a more accurate semantic code for each query point and utilizes the semantic codes to uniformize the structure of reconstructed 3D head meshes with the standard 3DMM. Since our reconstructed 3D head meshes have the same structure as the 3DMM, 3DMM-based animation approaches can be easily transferred to animate our reconstructed 3D heads. As a result, SeIF can reconstruct high-fidelity, artifact-less, and animatable 3D heads from single images of individuals with diverse ages, genders, races, and facial expressions. Quantitative and qualitative experimental results on seven datasets show that SeIF outperforms existing state-of-the-art methods by a large margin.

![](https://github.com/starVisionTeam/SeIF/blob/master/lib/data/final2.gif)


This repository includes code and some reconstruction results of our paper. Note that all of the code and results can be only used for research purposes.
***

## Citation

Please consider citing the paper if you find the code useful in your research.
```
@article{SeIF_TMM2024,
  author = {Liu, Leyuan and Liu, Xu and Sun, Jianchi and Gao, Changxin and Chen, Jingying},
  journal = IEEE Transactions on Multimedia (TMM), 
  title = {SeIF: Semantic-constrained Deep Implicit Function for Single-image 3D Head Reconstruction}, 
  year = {2024},
  pages = {1-15},
  DOI = {10.1109/TMM.2024.3405721}
}
```
The early access version can be downloaded from [https://ieeexplore.ieee.org/document/10539280](https://ieeexplore.ieee.org/document/10539280)


## Demos
This code has been tested with PyTorch 1.4.0 and CUDA 10.1. on Ubuntu 18.04.
We provide test code and pre-trained models. After installing the packages required for the project, run the following code:</br>
```objpython
python main_test.py
```
This code inputs the picture in the demo folder and outputs the 3D head model reconstructed from this picture. The corresponding results are saved in a demo folder. Such as input: demo/1.png, output: demo/1.obj. You can download [meshlab](https://www.meshlab.net/#download) to view 3D models.</br>
</br>

## Pre-trained Normal Estimation Model
We put the pre-trained normal estimation Code in
```
https://github.com/sama0626/Head_normal/tree/master
```
The pre-trained normal estimation model can be downloaded from
```
https://pan.baidu.com/s/1ay8g8qRWZ0Uw_i2kXOAWWg
```
Access code: SeIF

***
## Results
We compare our method with eight state-of-the-art methods that have publicly released codes: [PRNet](https://github.com/yfeng95/PRNet), [Deep3D](https://github.com/microsoft/Deep3DFaceReconstruction), [3DDFAV2](https://github.com/cleardusk/3DDFA_V2), [MGCNet](https://github.com/jiaxiangshang/MGCNet), [DECA](https://github.com/yfeng95/DECA), [Hifi3dface](https://github.com/tencent-ailab/hifi3dface), [MICA](https://github.com/Zielon/MICA), and [EMOCA](https://github.com/radekd91/emoca).</br>
![](https://github.com/starVisionTeam/SeIF/blob/master/lib/data/1-more.png)
<p align="center">Figure 1: Reconstruction results produced by different methods on the FamousFace datasets.</p>

![](https://github.com/starVisionTeam/SeIF/blob/master/lib/data/2-more.png)
<p align="center">Figure 2: Reconstruction results produced by different methods on the KidFace datasets.</p>

![](https://github.com/starVisionTeam/SeIF/blob/master/lib/data/3-more.png)
<p align="center">Figure 3: Reconstruction results produced by different methods on the FaceScape datasets.</p>

![](https://github.com/starVisionTeam/SeIF/blob/master/lib/data/4-more.png)
<p align="center">Figure 4: Reconstruction results produced by different methods on the CoMa datasets.</p>

![](https://github.com/starVisionTeam/SeIF/blob/master/lib/data/5-more.png)
<p align="center">Figure 5: Reconstruction results produced by different methods on the HeadSpace datasets.</p>

![](https://github.com/starVisionTeam/SeIF/blob/master/lib/data/6-more.png)
<p align="center">Figure 6: Reconstruction results produced by different methods on the FaceVerse datasets.</p>

![](https://github.com/starVisionTeam/SeIF/blob/master/lib/data/7-more.png)
<p align="center">Figure 7: Reconstruction results produced by different methods on the I3DMM datasets.</p>

![](https://github.com/starVisionTeam/SeIF/blob/master/lib/data/consistency.png)
<p align="center">Figure 8: Reconstructed 3D heads for the same subjects across different facial expressions.</p>

### Animation
![](https://github.com/starVisionTeam/SeIF/blob/master/lib/data/animationFinal3.png)
<p align="left">Figure 9: Results of model animation on the in-the-wild FamousFace and kidFace datasets. Given the pose and expression parameters estimated from the target
image, we can easily animate our reconstructed mesh to “copy” the jaw pose and expression of the target images using FLAME.</p>


## 3D Printed Works
![](https://github.com/starVisionTeam/SeIF/blob/master/lib/data/3Dprint-new.gif)

<p align="left"> We printed several reconstructed heads using a toy-grade 3D printer (purchased for approximately 100 USD). The flaws on the 3D-printed works were mainly caused by the low precision of the printer. By using a higher-precision 3D printer, we can avoid these flaws.</p>

## Acknowledgements
We benefit from external sources: [DECA](https://github.com/yfeng95/DECA), and [FLAME](https://github.com/soubhiksanyal/FLAME_PyTorch). We thank the authors for their great job!

***

## Contact
If you have any trouble when using this repo, please do not hesitate to send an E-mail to Xu Liu (xliu@mails.ccnu.edu.cn) or Leyuan Liu (lyliu@mail.ccnu.edu.cn).
