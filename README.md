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
We compare our method with eight state-of-the-art methods that have publicly released codes: PRNet, Deep3D, 3DDFAV2, MGCNet, DECA, Hifi3dface, MICA, and EMOCA.</br>
![](https://github.com/starVisionTeam/SeIF/blob/master/lib/data/1-more.png)
<p align="center">Figure: Reconstruction results produced by different methods on the FamousFace datasets.</p>
![](https://github.com/starVisionTeam/SeIF/blob/master/lib/data/2-more.png)
<p align="center">Figure: Reconstruction results produced by different methods on the KidFace datasets.</p>

