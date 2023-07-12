## Collections of Paper Focused on LMM4Vision or ViTs.

Welcome to open issues or pull requests (recommended)

### Awesome-LMM4Vision

- Preliminary: Visual problem solving：“**What is where?**“
  - Vision-only
    - Classification>What
    - Detection->What+Where(object level)
    - Segmentation->What+Where(pixel level)
    - Restoration->What(pixel level)
  - Vision-Language
    - Grounding->**something** at **where**
    - Caption->what+where+**action**
    - Generation*->what+where+action
  - Vision-others

- [SPAE] Semantic Pyramid AutoEncoder for Multimodal Generation with Frozen LLMs.
  
  - model non-linguistic modality as a language sequence that LLMs can comprehend
  - In contrast to the majority of VQ-VAE approaches, SPAE maps to an interpretable **discrete latent space**, i.e., words.
  - Dilation subsampler selects the positions for quantization
  
- [[Unified-IO](https://arxiv.org/abs/2206.08916)] A Unified Model for Vision, Language, and Multi-Modal Tasks.[[demo](https://unified-io.allenai.org/)]

  - 复杂的视觉任务常常因为输出的多样性而难以统一，但借助自然语言的灵活性或许可以提供一种可行的措施。
  - 借助LLMs进行多个视觉任务的共同学习（如[DetCLIPv2](https://arxiv.org/pdf/2304.04514.pdf)**交替使用**不同类型的数据如Detection、Grounding和Image-Text pair进行学习，分别赋予模型定位和对更广泛概念的认识，从而实现OVD），飞轮效应。
  - 另外一个想法则是，选择一个足够具备挑战性的任务（例如[Hiera](https://arxiv.org/abs/2306.00989)则选取Mask Image Model来作为代表性任务训练自己所提出的transformer模型）

  - Insights：

- [[VisionLLM](https://arxiv.org/abs/2305.11175)] Large Language Model is also an Open-Ended Decoder for Vision-Centric Tasks[[demo](https://igpt.opengvlab.com/)]

  - Insights: considers images as a kind of foreign language and converts them into token representations.

  - Summary:分别使用常规的图像编码器与文本编码器提取多层次的视觉特征和文本特征，通过交叉注意力机制将文本特征注入图像特征（即Language-Guided Image Tokenizer），再借助<embedding, position>形式的Image Token作为Query既提取语义信息也提取位置信息。在送入LLMs的解码器之前，为解决类别和位置token数量不足的问题，将<class>token与<position>token进行增广，前者丰富了视觉任务的输出而后者则将回归问题实际上转换为离散的分类问题。最后，通过统一输出格式示例来限制LLMs的输出是高度与任务契合的

    `(e.g., “<cls> <x1> <y1> <x2> <y2>” for object detection, “<bos>” for image captioning)`

  - 思考：该工作并没有提出任何新的模型，而是在现有预训练的模型基础上将图像与提示文本（与任务高度相关）进行融合，并在利用LLMs的decoder输出前统一格式。

  - 训练代价：4$\times$8 A100，可更新参数是Backbone和D-DETR和LLM的LoRA的少量参数

  - Rating: 创新性不高，所做的视觉任务不够多。但从本文章认识到计算机视觉任务可以分为Vision-only与Vision-Language，同时prompts中的<image>能否？

  ![](./figs/LLMVision.png)

### Awesome-Vision-Transformers

#### CVPR 2023

- [[Castiling-ViT](https://openaccess.thecvf.com/content/CVPR2023/papers/You_Castling-ViT_Compressing_Self-Attention_via_Switching_Towards_Linear-Angular_Attention_at_Vision_CVPR_2023_paper.pdf)] Compressing Self-Attention via Switching Towards Linear-Angular Attention at Vision Transformer Inference. [[code](https://www.haoranyou.com/castling-vit/)]
  - Linear-Angular Attention to measuring spectral similarity

- [[HGFormer](https://openaccess.thecvf.com/content/CVPR2023/papers/Ding_HGFormer_Hierarchical_Grouping_Transformer_for_Domain_Generalized_Semantic_Segmentation_CVPR_2023_paper.pdf)]Hierarchical Grouping Transformer for Domain Generalized Semantic Segmentation [[code](https://github.com/dingjiansw101/HGFormer)]

