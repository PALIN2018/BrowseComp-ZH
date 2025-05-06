
# 🧭 BrowseComp-ZH：面向中文网络环境的大模型网页浏览能力基准测试

**BrowseComp-ZH** 是首个专为评估大型语言模型（LLMs）在中文网络生态中检索与推理能力而设计的高难度基准测试。受 [BrowseComp (Wei 等, 2025)](https://openai.com/index/browsecomp/) 启发，本项目针对中文信息环境构建了复杂的多跳检索与推理任务，模型需应对平台碎片化、语言特性及内容审查等多重挑战。

📄 [项目论文链接（arXiv）](https://arxiv.org/pdf/2504.19314)

## 👥 作者

Peilin Zhou, Bruce Leon, Xiang Ying, Can Zhang, Yifan Shao, Qichen Ye, Dading Chong, Zhiling Jin, Chenxuan Xie, Meng Cao, Yuxin Gu, Sixin Hong, Jing Ren, Jian Chen, Chao Liu, Yining Hua

## 🌟 项目亮点

- 🔍 **原生中文构建，打破语言局限**：所有问题由中文母语标注员直接设计，真实反映中文网络检索挑战，避免翻译偏差。
- 🧩 **高难度逆向设计与多跳推理**：每道题目均从已知答案逆向构建，融合时间、类别、描述等多维约束，要求跨平台、多步检索与推理。
- 🌐 **三引擎验证与双阶段质量控制**：所有问题经百度、必应（中国版）、谷歌三引擎交叉验证，并通过人工与机器双重审核，确保检索难度和答案唯一性。
- 🤖 **全面基准测试，推动智能体发展**：覆盖 20+ 种开源、闭源及检索增强型模型，系统揭示当前 LLM 在中文多跳检索与推理中的核心瓶颈，助力检索增强智能体系统演进。
## 📁 仓库结构

```
BrowseComp-ZH/
├── data/
│   ├── browsecomp-zh-encrypted.xlsx   # 加密后的数据集
│   └── browsecomp-zh-decrypt.py       # 数据解密脚本
├── images/                            # 图表与可视化资源
├── paper/                             # 论文及附录
├── README.md
└── requirements.txt
```

## 🔐 数据访问
BrowseComp-ZH 数据集包含 **289 个多跳检索推理问题**，所有问题均以中文撰写。

为避免数据被滥用于大型语言模型的预训练，同时保护任务的原创性与评测有效性，browsecomp-zh-encrypted.xlsx 文件已加密。访问原始数据集请执行：

### 解密数据集
```bash
python data/browsecomp-zh-decrypt.py --input data/browsecomp-zh-encrypted.xlsx --output data/browsecomp-zh-decrypted.xlsx
```
系统将提示输入嵌入式密码（canary token）。

## 🏆 模型表现概览

BrowseComp-ZH 测试了 20+ 个开源、闭源及智能检索增强型系统。以下是各模型在数据集上的表现概览：

| 模型                     | 类别             | 是否具备推理 | 是否具备浏览 | 准确率   | 校准误差 (%) | 所属企业   |
|--------------------------|------------------|--------------|--------------|----------|--------------|------------|
| DeepSeek-V3              | 开源模型         | 否           | 否           | 8.7%     | 72           | DeepSeek   |
| DeepSeek-R1              | 开源模型         | 是           | 否           | 23.2%    | 59           | DeepSeek   |
| Qwen2.5-72B-Instruct     | 开源模型         | 否           | 否           | 6.6%     | 62           | Alibaba    |
| Qwen3-235B-A22B (Non-Thinking)| 开源模型         | 否           | 否           | 8.0%    | 80           | Alibaba    |
| Qwen3-235B-A22B (Thinking)| 开源模型         | 是           | 否           | 13.2%    | 67           | Alibaba    |
| QwQ-32B                  | 开源模型         | 是           | 否           | 11.1%    | 64           | Alibaba    |
| LlaMa4                   | 开源模型         | 否           | 否           | 4.8%     | 70           | Meta       |
| GPT4o                    | 闭源模型         | 否           | 否           | 6.2%     | 73           | OpenAI     |
| O1                       | 闭源模型         | 是           | 否           | 29.1%    | 52           | OpenAI     |
| O4-mini                  | 闭源模型         | 是           | 否           | 15.2%    | 42           | OpenAI     |
| Claude-3.5-Sonnet        | 闭源模型         | 否           | 否           | 5.5%     | 78           | Anthropic  |
| Claude-3.7-Sonnet        | 闭源模型         | 是           | 否           | 17.7%    | 71           | Anthropic  |
| Gemini-2.0-Flash         | 闭源模型         | 否           | 否           | 6.9%     | 74           | Google     |
| Gemini-2.5-Pro           | 闭源模型         | 是           | 否           | 27.3%    | 59           | Google     |
| Qwen2.5-MAX              | 闭源模型         | 否           | 否           | 7.6%     | 78           | Alibaba    |
| OpenAI DeepResearch      | AI搜索系统       | -            | 是           | **42.9%**| 9            | OpenAI     |
| Grok3 (Research)         | AI搜索系统       | -            | 是           | 12.9%    | 39           | xAI        |
| Perplexity (Research)    | AI搜索系统       | -            | 是           | 22.6%    | 53           | Perplexity |
| Doubao (Deep Search)     | AI搜索系统       | -            | 是           | 26.0%    | 61           | ByteDance  |
| Doubao (Standard)        | AI搜索系统       | -            | 是           | 18.7%    | 37           | ByteDance  |
| Kimi (Deep Think)        | AI搜索系统       | -            | 是           | 8.0%     | 58           | Moonshot   |
| Yuanbao (Hunyuan Model)  | AI搜索系统       | -            | 是           | 12.2%    | 56           | Tencent    |
| DeepSeek (Deep Think)    | AI搜索系统       | -            | 是           | 7.6%     | 65           | DeepSeek   |
| DeepSeek (Standard)      | AI搜索系统       | -            | 是           | 4.8%     | 66           | DeepSeek   |


## 📊 模型表现总结

- 📉 **绝大多数语言模型（LLMs）在 BrowseComp-ZH 上准确率低于 10%。**
- 🧠 **具备推理能力的模型准确率显著提升。**
- 🔍 **检索增强型智能体系统表现优于纯语言模型。**
- 🔄 **单轮检索系统普遍难以胜任任务，需要多轮检索与推理。**
- 📈 **模型校准误差与检索推理能力相关，影响最终性能表现。**

## 📎 引用格式

如果您在研究中使用了 BrowseComp-ZH，请引用如下：

```bibtex
@article{zhou2025browsecomp,
  title={BrowseComp-ZH: Benchmarking Web Browsing Ability of Large Language Models in Chinese},
  author={Zhou, Peilin and Leon, Bruce and Ying, Xiang and Zhang, Can and Shao, Yifan and Ye, Qichen and Chong, Dading and Jin, Zhiling and Xie, Chenxuan and Cao, Meng and others},
  journal={arXiv preprint arXiv:2504.19314},
  year={2025}
}
```

## 🤝 联系与贡献

欢迎提问、建议或贡献代码！  
请通过 issue 提交问题，或联系 [@PALIN2018](https://github.com/PALIN2018) 与我们交流。


## 🛡️ 许可协议

BrowseComp-ZH 遵循 [MIT License](./LICENSE) 发布。  
**数据集仅限于学术研究使用**，请勿将其直接应用于医疗、法律等敏感领域决策中。