# 🧭 BrowseComp-ZH: Benchmarking the Web Browsing Ability of Large Language Models in Chinese

[🇨🇳 中文版项目介绍 (Chinese)](./README-ZH.md)

**BrowseComp-ZH** is the first high-difficulty benchmark specifically designed to evaluate the real-world web browsing and reasoning capabilities of large language models (LLMs) in the Chinese information ecosystem. Inspired by [BrowseComp (Wei et al., 2025)](https://openai.com/index/browsecomp/), BrowseComp-ZH targets the unique linguistic, structural, and retrieval challenges of the Chinese web, including fragmented platforms, implicit linguistic patterns, and content censorship.

📄 [Paper Link（arXiv）](https://arxiv.org/pdf/2504.19314)

## 👥 Authors

Peilin Zhou, Bruce Leon, Xiang Ying, Can Zhang, Yifan Shao, Qichen Ye, Dading Chong, Zhiling Jin, Chenxuan Xie, Meng Cao, Yuxin Gu, Sixin Hong, Jing Ren, Jian Chen, Chao Liu, Yining Hua

## 🌟 Key Features

- 🔍 **Native Chinese Construction**: All questions, retrieval chains, and browsing steps are authored directly in Chinese by experts to avoid translation artifacts and ensure authentic search difficulty.
- 🧩 **Reverse-Engineered Multi-Hop Queries**: Each task starts from a known factual answer and is crafted with multiple constraints (e.g., time, entity type, description) to ensure high retrieval difficulty and answer uniqueness.
- 🌐 **Tri-Engine Validation and Dual-Stage Quality Control**: All questions are verified across Baidu, Bing (China), and Google; a two-stage human-in-the-loop protocol filters out easily retrievable or ambiguous samples.
- 🤖 **Comprehensive Benchmarking**: 20+ systems—including open-source LLMs, closed-source APIs, and agentic search systems—are evaluated to diagnose browsing and reasoning capabilities across different architectures.

## 📁 Repository Structure

```
BrowseComp-ZH/
├── data/
│   ├── browsecomp-zh-encrypted.xlsx   # Encrypted dataset
│   └── browsecomp-zh-decrypt.py       # Decryption script
├── images/                            # Visualizations and charts
├── paper/                             # Paper and supplementary 
├── README.md
└── requirements.txt
```

## 🔐 Dataset Access

The BrowseComp-ZH dataset contains **289 complex multi-hop retrieval and reasoning questions**, spanning 11 domains including Film & TV, Technology, Medicine, and History.

To prevent unauthorized pretraining and preserve the evaluation value of the dataset, all samples are encrypted.  
To decrypt the dataset:

```bash
python data/browsecomp-zh-decrypt.py --input data/browsecomp-zh-encrypted.xlsx --output data/browsecomp-zh-decrypted.xlsx
```
You will be prompted for a canary token embedded within the file.

## 🏆 Model Performance Overview

| Model                    | Category          | Reasoning | Browsing | Accuracy | Calibration Error (%) | Enterprise |
|---------------------------|-------------------|-----------|----------|----------|------------------------|------------|
| DeepSeek-V3               | Open-Source       | No        | No       | 8.7%     | 72                     | DeepSeek   |
| DeepSeek-R1               | Open-Source       | Yes       | No       | 23.2%    | 59                     | DeepSeek   |
| Qwen2.5-72B-Instruct      | Open-Source       | No        | No       | 6.6%     | 62                     | Alibaba    |
| QwQ-32B                   | Open-Source       | Yes       | No       | 11.1%    | 64                     | Alibaba    |
| Qwen3-235B-A22B (Non-Thinking)| Open-Source         | No           | No           | 8.0%    | 80           | Alibaba    |
| Qwen3-235B-A22B (Thinking)| Open-Source         | Yes           | No           | 13.2%    | 67           | Alibaba    |
| LlaMa4                    | Open-Source       | No        | No       | 4.8%     | 70                     | Meta       |
| GPT4o                     | Closed-Source     | No        | No       | 6.2%     | 73                     | OpenAI     |
| O1                        | Closed-Source     | Yes       | No       | 29.1%    | 52                     | OpenAI     |
| O4-mini                   | Closed-Source     | Yes       | No       | 15.2%    | 42                     | OpenAI     |
| Claude-3.5-Sonnet         | Closed-Source     | No        | No       | 5.5%     | 78                     | Anthropic  |
| Claude-3.7-Sonnet         | Closed-Source     | Yes       | No       | 17.7%    | 71                     | Anthropic  |
| Gemini-2.0-Flash          | Closed-Source     | No        | No       | 6.9%     | 74                     | Google     |
| Gemini-2.5-Pro            | Closed-Source     | Yes       | No       | 27.3%    | 59                     | Google     |
| Qwen2.5-MAX               | Closed-Source     | No        | No       | 7.6%     | 78                     | Alibaba    |
| OpenAI DeepResearch       | AI Search Product | -         | Yes      | **42.9%**| 9                      | OpenAI     |
| Grok3 (Research)          | AI Search Product | -         | Yes      | 12.9%    | 39                     | xAI        |
| Perplexity (Research)     | AI Search Product | -         | Yes      | 22.6%    | 53                     | Perplexity |
| Doubao (Deep Search)      | AI Search Product | -         | Yes      | 26.0%    | 61                     | ByteDance  |
| Doubao (Standard)         | AI Search Product | -         | Yes      | 18.7%    | 37                     | ByteDance  |
| Kimi (Deep Think)         | AI Search Product | -         | Yes      | 8.0%     | 58                     | Moonshot   |
| Yuanbao (Hunyuan Model)   | AI Search Product | -         | Yes      | 12.2%    | 56                     | Tencent    |
| DeepSeek (Deep Think)     | AI Search Product | -         | Yes      | 7.6%     | 65                     | DeepSeek   |
| DeepSeek (Standard)       | AI Search Product | -         | Yes      | 4.8%     | 66                     | DeepSeek   |

## 📊 Key Findings

- 📉 **Most standalone LLMs achieve less than 10% accuracy on BrowseComp-ZH**, reflecting the benchmark’s difficulty.
- 🧠 **Models with explicit reasoning capabilities consistently outperform their non-reasoning counterparts** (e.g., DeepSeek-R1 vs. DeepSeek-V3, Claude-3.7 vs. Claude-3.5).
- 🔍 **Retrieval-augmented systems significantly outperform pure LLMs**, with DeepResearch achieving the highest accuracy (42.9%).
- 🔄 **Multi-hop retrieval pipelines are critical**: Single-shot retrieval systems (e.g., DeepSeek, Kimi) struggle to meet task complexity.
- 📈 **Calibration error correlates with retrieval-reasoning effectiveness**, highlighting challenges in confidence estimation during browsing.

## 📎 Citation

If you use BrowseComp-ZH in your research, please cite:

```bibtex
@article{zhou2025browsecomp,
  title={BrowseComp-ZH: Benchmarking Web Browsing Ability of Large Language Models in Chinese},
  author={Zhou, Peilin and Leon, Bruce and Ying, Xiang and Zhang, Can and Shao, Yifan and Ye, Qichen and Chong, Dading and Jin, Zhiling and Xie, Chenxuan and Cao, Meng and others},
  journal={arXiv preprint arXiv:2504.19314},
  year={2025}
}
```

## 🤝 Contact & Contribution

We welcome questions, suggestions, and contributions!  
Please open an issue or contact [@PALIN2018](https://github.com/PALIN2018).

## 🛡️ License

BrowseComp-ZH is released under the [MIT License](./LICENSE).  
**The dataset is intended solely for academic research purposes and must not be used for sensitive or high-stakes decision-making.**
