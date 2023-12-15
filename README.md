# <p align="center">SeIF: Semantic-constrained Deep Implicit Function for Single-image 3D Head Reconstruction</p>
#### <p align="center">Leyuan Liu, Xu Liu, Jianchi Sun, Changxin Gao, Jingying Chen</p>
***
![](https://github.com/starVisionTeam/SeIF/blob/master/lib/data/final.gif)

This repository includes Training code and testing code of our paper. It also contains some reconstruction results. Note that all of the results can be only used for research purposes.
***
## Demos
We provide test code and pre-trained models. After installing the packages required for the project, run the following code:</br>
```objpython
python main_test.py
```
This code inputs the picture in the demo folder and outputs the 3D head model reconstructed from this picture. The corresponding results are saved in a demo folder. Such as input: demo/1.png, output: demo/1.obj. You can download [meshlab](https://www.meshlab.net/#download) to view 3D models.</br>
</br>
The train dataset is available for non-commercial or research use only, we will publish the training code and training data through email application. My email is xliu@mails.ccnu.edu.cn.
</br>
***
## MORE VISUALIZATION RESULTS
We compare our method with eight state-of-the-art methods that have publicly released codes: [PRNet](https://github.com/yfeng95/PRNet), [Deep3D](https://github.com/microsoft/Deep3DFaceReconstruction), [3DDFAV2](https://github.com/cleardusk/3DDFA_V2), [MGCNet](https://github.com/jiaxiangshang/MGCNet), [DECA](https://github.com/yfeng95/DECA), [Hifi3dface](https://github.com/tencent-ailab/hifi3dface), [MICA](https://github.com/Zielon/MICA), and [EMOCA](https://github.com/radekd91/emoca).</br>
![](https://github.com/starVisionTeam/SeIF/blob/master/lib/data/1-more.png)
<p align="center">Figure: Reconstruction results produced by different methods on the FamousFace datasets.</p>

![](https://github.com/starVisionTeam/SeIF/blob/master/lib/data/2-more.png)
<p align="center">Figure: Reconstruction results produced by different methods on the KidFace datasets.</p>

![](https://github.com/starVisionTeam/SeIF/blob/master/lib/data/3-more.png)
<p align="center">Figure: Reconstruction results produced by different methods on the FaceScape datasets.</p>

![](https://github.com/starVisionTeam/SeIF/blob/master/lib/data/4-more.png)
<p align="center">Figure: Reconstruction results produced by different methods on the CoMa datasets.</p>

![](https://github.com/starVisionTeam/SeIF/blob/master/lib/data/5-more.png)
<p align="center">Figure: Reconstruction results produced by different methods on the HeadSpace datasets.</p>

![](https://github.com/starVisionTeam/SeIF/blob/master/lib/data/6-more.png)
<p align="center">Figure: Reconstruction results produced by different methods on the FaceVerse datasets.</p>

![](https://github.com/starVisionTeam/SeIF/blob/master/lib/data/7-more.png)
<p align="center">Figure: Reconstruction results produced by different methods on the I3DMM datasets.</p>
