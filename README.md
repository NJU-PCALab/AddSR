## AddSR: Accelerating Diffusion-based Blind Super-Resolution with Adversarial Diffusion Distillation

<a href='https://arxiv.org/abs/2404.01717'><img src='https://img.shields.io/badge/arXiv-2404.01717-b31b1b.svg'></a> &nbsp;&nbsp; [![ProjectPage](https://img.shields.io/badge/ProjectPage-AddSR-orange.svg)](https://nju-pcalab.github.io/projects/AddSR/) &nbsp;&nbsp; ![visitors](https://visitor-badge.laobi.icu/badge?page_id=CSRuiXie.AddSR)

[Rui Xie](https://github.com/CSRuiXie)<sup>1</sup> | [Ying Tai](https://tyshiwo.github.io/index.html)<sup>2</sup> | [Kai Zhang](https://cszn.github.io/)<sup>2</sup> | [Zhenyu Zhang](https://jessezhang92.github.io/)<sup>2</sup> | [Jun Zhou](https://scholar.google.com/citations?hl=zh-CN&user=w03CHFwAAAAJ)<sup>1</sup> | [Jian Yang](https://scholar.google.com.hk/citations?user=6CIDtZQAAAAJ&hl=zh-CN)<sup>3</sup>

<sup>1</sup>Southwest University, <sup>2</sup>Nanjing University, <sup>3</sup>Nanjing University of Science and Technology. 


### 💬 News
- **2024.04.09** 🚀 Release the pretrained model and testing code.


### 📌 TODO
- ✅ Release the pretrained model
- [ ] Release the training code

## 🔎 Method Overview
![AddSR](figs/framework.png)

## 📷 Results Display
[<img src="figs/flower.png" height="320px"/>](https://imgsli.com/MjUyNTc5) [<img src="figs/building.png" height="320px"/>](https://imgsli.com/MjUyNTkx) 
[<img src="figs/nature.png" height="320px"/>](https://imgsli.com/MjUyNTgx) [<img src="figs/human.png" height="320px"/>](https://imgsli.com/MjUyNTky)



![AddSR](figs/real_world.png)

## ⚙️ Dependencies and Installation
```
## git clone this repository
git clone https://github.com/NJU-PCALab/AddSR.git
cd AddSR

# create an environment with python >= 3.8
conda create -n addsr python=3.8
conda activate addsr
pip install -r requirements.txt
```

## 🚀 Inference
#### Step 1: Download the pretrained models
- Download the pretrained SD-2-base models from [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-base).
- Download the AddSR models from [GoogleDrive](https://drive.google.com/file/d/19dMAc4mzFSSfU23y5g44v3nZS-edubGw/view?usp=sharing)
- Download the DAPE models from [GoogleDrive](https://drive.google.com/drive/folders/12HXrRGEXUAnmHRaf0bIn-S8XSK4Ku0JO?usp=drive_link)

You can put the models into `preset/`.

#### Step 2: Prepare testing data
You can put the testing images in the `preset/datasets/test_datasets`.

#### Step 3: Running testing command
```
python test_addsr.py \
--pretrained_model_path preset/models/stable-diffusion-2-base \
--prompt '' \
--addsr_model_path preset/models/addsr \
--ram_ft_path preset/models/DAPE.pth \
--image_path preset/datasets/test_datasets \
--output_dir preset/datasets/output \
--start_point lr \
--num_inference_steps 4 \
--PSR_weight 0.5
```

## ❤️ Acknowledgments
This project is based on [SeeSR](https://github.com/cswry/SeeSR), [diffusers](https://github.com/huggingface/diffusers), [BasicSR](https://github.com/XPixelGroup/BasicSR), [ADD](https://arxiv.org/abs/2311.17042) and [StyleGAN-T](https://github.com/autonomousvision/stylegan-t). Thanks for their awesome works.

## 🎓Citations
If our project helps your research or work, please consider citing our paper:

```
@misc{xie2024addsr,
      title={AddSR: Accelerating Diffusion-based Blind Super-Resolution with Adversarial Diffusion Distillation}, 
      author={Rui Xie and Ying Tai and Kai Zhang and Zhenyu Zhang and Jun Zhou and Jian Yang},
      year={2024},
      eprint={2404.01717},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
