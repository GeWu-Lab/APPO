<!-- <p align="center">
    <img src="assets/crab.jpeg" width="150" style="margin-bottom: 0.2;"/>
<p> -->


<h3 align="center"> (CVPR'26) <a href="" style="color:#9C276A">
APPO: Attention-guided Perception Policy Optimization for Video Reasoning
</a></h3>


<h5 align="center"> 🚀🚀 Welcome to the repo of APPO! If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏 </h2>

<h5 align="center">


[![hf_checkpoint](https://img.shields.io/badge/🤗-APPO-9C276A.svg)](https://huggingface.co/CserDu123/ARPO) [![arXiv](https://img.shields.io/badge/Arxiv-2503.13068-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2602.23823)<br>

<img src="assets/teaser.png" width="800" />

We present APPO, the Attention-guided Perception Policy Optimization algorithm that enhances model’s fine-grained perception
ability through reasoning. The core idea behind APPO is to optimize those tokens from different responses that primarily focus on the
same crucial video frames (called intra-group perception tokens), resulting in fine-grained token level reward signals. Left: The illustration
of APPO algorithm. The intra-group perception tokens are defined as those tokens from different responses that primarily focus on the
same crucial video frame. The perception tokens within each group are optimized with different learning intensities. Right: Experimental
results on multiple video benchmarks demonstrate APPO achieves overall performance improvement compared with GRPO and DAPO.

<img src="assets/perception-reasoning-curves.png" width="800" />
The Perception-Reasoning curves on SEED-Bench-R1 [4] and Perception-Test [20] benchmarks, quantifying the impact of
perception vs. reasoning ability on overall performance. Each point in the curve represents the performance achieved by combining
specific perception and reasoning ability. In particular, we first prompted four perception models with progressively enhanced abilities
(including Qwen2.5-VL-3/7/32B [1] and Gemini-2.0-flash [5]) to describe video content in detail. Subsequently, the other four reasoning
models with varying capabilities (including Qwen3-4/8B, Qwen3-235-A22B-thinking [30], and OpenAI-o3 [12]) were used to think and
answer questions based on the descriptions provided by each perception model, respectively, yielding 4 × 4 cross-combination results.
(a) For SEED-Bench-R1 benchmark, we evaluate on 2K Level-1 samples. (b) For Perception Test benchmark, we randomly select 1K
samples from different videos for evaluation. (c) The performance comparison of GRPO, DAPO and our APPO on SEED-Bench-R1
benchmark across different scales models, demonstrating the significant improvements brought by enhanced perception.



## 📰 News

* **[2026.03.18]**   Release training and evaluation codes of Crab.
* **[2026.02.20]**   APPO has been accepted to CVPR 2026.


<!-- <div align="center"><video src="https://www.youtube.com/watch?v=O57xewSvOj4" ></div> -->


## 🛠️ Requirements and Installation
Basic Dependencies:
* Python == 3.10
* trl == 0.23.1
* transformers == 4.52.3
* deepspeed == 0.16.4
* accelerate == 1.8.1

Install required packages:
```bash
git clone git@github.com:GeWu-Lab/APPO.git
cd APPO
pip install -r requirements.txt
```

## 🗝️ RL Training
  
Training on Seed-Bench-R1 or Video-R1 data:
```bash
bash scripts/qwen2_5_vl_rl.sh
```

Training on NeXT-GQA data:
```bash
bash scripts/qwen2_5_vl_rl_tvg.sh
``` 


## 📑 Citation
If you find APPO useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{du2026appo,
  title={APPO: Attention-guided Perception Policy Optimization for Video Reasoning},
  author={Du, Henghui and Zhou, Chang and Chen, Xi and Hu, Di},
  journal={arXiv preprint arXiv:2602.23823},
  year={2026}
}
```

## 🔒 License

This project is released under the Apache 2.0 license as found in the LICENSE file.
Please get in touch with us if you find any potential violations.
