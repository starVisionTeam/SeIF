# <p align="center">SeIF: Semantic-constrained Deep Implicit Function for Single-image 3D Head Reconstruction</p>
#### <p align="center">Leyuan Liu, Xu Liu, Jianchi Sun, Changxin Gao, Jingying Chen</p>
***
  Various applications require 3D avatars that are realistic, artifact-free, and animatable. However, traditional 3D morphable models (3DMMs) produce animatable 3D heads but fail to capture accurate geometries and details, while existing deep implicit functions have been shown to achieve realistic reconstructions but suffer from artifacts and struggle to yield 3D heads that are easy to animate. To reconstruct **high-fidelity**, **artifact-less**, and [**animatable**](### Animation) 3D heads from single images, we leverage dense semantics to bridge the best properties of 3DMMs and deep implicit functions and propose SeIF---a semantic-constrained deep implicit function. First, SeIF derives dense semantics from a standard 3DMM (e.g., FLAME) and samples a semantic code for each query point in the query space to provide a soft constraint to the deep implicit function. The reconstruction results show that this semantic constraint does not weaken the powerful representation ability of the deep implicit function while significantly suppressing artifacts. Second, SeIF predicts a more accurate semantic code for each query point and utilizes the semantic codes to uniformize the structure of reconstructed 3D head meshes with the standard 3DMM. Since our reconstructed 3D head meshes have the same structure as the 3DMM, 3DMM-based animation approaches can be easily transferred to animate our reconstructed 3D heads. As a result, SeIF can reconstruct high-fidelity, artifact-less, and animatable 3D heads from single images of individuals with diverse ages, genders, races, and facial expressions. Quantitative and qualitative experimental results on seven datasets show that SeIF outperforms existing state-of-the-art methods by a large margin.

![](https://github.com/starVisionTeam/SeIF/blob/master/lib/data/final2.gif)

This repository includes code and some reconstruction results of our paper. Note that all of the code and results can be only used for research purposes.
***
## Demos
This code has been tested with PyTorch 1.4.0 and CUDA 10.1. on Ubuntu 18.04.
We provide test code and pre-trained models. After installing the packages required for the project, run the following code:</br>
```objpython
python main_test.py
```
This code inputs the picture in the demo folder and outputs the 3D head model reconstructed from this picture. The corresponding results are saved in a demo folder. Such as input: demo/1.png, output: demo/1.obj. You can download [meshlab](https://www.meshlab.net/#download) to view 3D models.</br>
</br>

***
## VISUALIZATION RESULTS
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
<p align="center">Figure 9: Results of model animation on the in-the-wild FamousFace and kidFace datasets. Given the pose and expression parameters estimated from the target
image, we can easily animate our reconstructed mesh to “copy” the jaw pose and expression of the target images using FLAME.</p>


***
## Contact
The code is released. And we are still updating it. If you have any trouble when using this repo, please do not hesitate to send an E-mail to Xu Liu(xliu@mails.ccnu.edu.cn)
