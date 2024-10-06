<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
                Few-Shot Image Generation by <br> Conditional Relaxing Diffusion Inversion</h1>
<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://scholar.google.com/" target="_blank" style="text-decoration: none;">Yu Cao<sup>*</sup></a>&nbsp;,&nbsp;
    <a href="http://www.eecs.qmul.ac.uk/~sgg/" target="_blank" style="text-decoration: none;">Shaogang Gong<sup>&#8224</sup></a></br>
</p>
<p align='center' style="text-align:center;font-size:1.25em;">
Queen Mary University of London<br/>
</p>

<p align='left';>
<b>
<em>ECCV 2024, </em>
<em>Sun Sep 29th through Fri Oct 4th, 2024
at MiCo Milano.</em>
</b>
</p>

<p align='left' style="text-align:left;font-size:1.3em;">
<b>
    [<a href="https://yucao16.github.io/CRDI/" target="_blank" style="text-decoration: none;">Project Page</a>]&nbsp;&nbsp;
</b>
<b>
    [<a href="https://arxiv.org/pdf/2407.07249" target="_blank" style="text-decoration: none;">Paper</a>]&nbsp;&nbsp;
</b>
</p>

## Usage

### 1. Download Pre-trained Model
Download [FFHQ 256*256 ckpt of Guided Diffusion from ddpm-segmentation](https://github.com/yandex-research/ddpm-segmentation) to `checkpoints/ddpm/ffhq.pt`

### 2. Update Directory Paths
Modify the path in following files:
```
./datasets/babies_target/babies.csv
./scripts/fs_gradient_evaluate.py
```

### 3. Run Experiments
In the file `main.sh`, we provide commands to reproduce experiments results on Babies.
```
bash main.sh
```

## Bibtex
If you find this project useful in your research, please consider citing our paper:

```
@article{cao2024few,
  title={Few-Shot Image Generation by Conditional Relaxing Diffusion Inversion},
  author={Cao, Yu and Gong, Shaogang},
  journal={arXiv preprint arXiv:2407.07249},
  year={2024}
}
```

## Acknowledgments
Parts of this project page were adopted from the [Nerfies](https://nerfies.github.io/) page.

## Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
