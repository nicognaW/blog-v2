---
author: Nico
pubDatetime: 2023-03-22T14:20:22.253Z
title: 语言模型应用的高级工程实践
postSlug: advanced-engineering-practices-of-language-model-applications
featured: true
draft: false
tags:
  - AI
  - LLM
ogImage: ""
description: 本文简洁明了地介绍 OpenAI API 的基本使用方法，重点探讨在撰写本文时期，一些颇具前瞻性的高级工程实践，并概述当下开源大语言模型的发展状况。
---

## 目录

> _本文含有使用大语言模型生成的文本。_
>
> **本文将持续更新（通常在北京时间每日晚上 10 点左右）**

自 OpenAI 发布 ChatGPT 以来，随着 text-davinci-003、gpt-3.5-turbo 等先进模型的逐步推出，大语言模型在工程应用领域呈现出百花齐放的景象。本文将首先简洁明了地介绍 OpenAI API 的基本使用方法，随后重点探讨在撰写本文时期，一些颇具前瞻性的高级工程实践。此外，还将提供若干实用的参考案例，同时概述当下开源大语言模型的发展状况。

## OpenAI API 简介

### 大语言模型

Stephen Wolfram 老爷子做过一场[直播](https://www.youtube.com/watch?v=flXrLGPY3SU)来介绍、解释语言模型和大语言模型的知识与上下文，非常值得研究。

### ChatGPT 与 OpenAI API 的关系

ChatGPT 是 OpenAI 采用 GPT 系列模型，在对话场景中经过 fine-tune （调参）而研发出的一款产品。得益于其迅速崛起的知名度，人们将其底层运用的 gpt-3.5-turbo 模型称为 ChatGPT 模型，同时也有人将使用该模型的 Chat Completion API 称为 ChatGPT API。值得注意的是，OpenAI API 与 ChatGPT 这类面向消费者（C 端）的产品有所区别，是一套供开发人员使用的 API。

以下内容是使用 ChatGPT 生成的，对 OpenAI API 的介绍：

> OpenAI API 可以应用于几乎任何涉及自然语言理解或生成、代码或图像的任务。
>
> API 提供了不同功率级别的模型以适应不同任务，并支持自定义模型的微调。
>
> API 可用于各种任务，包括内容生成、语义搜索和分类、文本摘要、扩展、对话、创意写作、样式转移等。
>
> 设计提示（prompt）是如何“编程”模型的方法，可以应用于几乎任何任务，包括生成内容或代码、摘要、对话、创意写作、样式转移等。
>
> 模型将文本分解成单词或字符块，并以 token 的形式处理。API 支持不同价格和能力的模型，包括 GPT-4 和 GPT-3.5 Turbo 等。
>
> 在开始构建应用程序时，要记住使用策略。OpenAI 提供了不同的指南和示例库，供用户参考。

### Completion API

Completion API 是语言模型提供的最直观的能力之一，也是实际应用场景中最常用的 API。

其核心功能在于，阅读你提供的文本并猜测可能接在文本后面的内容。这种猜测的依据来源于模型所阅读过的数百亿文本。值得注意的是，与人们的直觉相反，语言模型并不会明确地进行逻辑思考，而是基于猜测来生成最为合理、最逼真的文本。

GPT 模型之所以能够令人惊叹地实现这一功能，归功于其庞大的训练数据规模。例如，GPT3 模型的训练数据集大小为 570 GB，而本文截至完成时的文件大小为 15 KB。换言之，GPT 所阅读的文本量相当于 3 千 8 百万篇本文长度的文章。

> 实际上 GPT3 模型早在 2020 年 5 月就发布了，我也曾经体验过一些早期的 GPT3 应用，有一些也是 chat bot 这样的形式，但与 ChatGPT 的效果相差甚远。主要是因为 OpenAI 在 GPT3 模型的基础上，加了[人类反馈的强化学习](https://openai.com/blog/chatgpt#methods)，来微调模型，使其迅速适应对话场景。

Completion API 分为 Text Completion 和 Chat Completion。其中，Text Completion 对应较早期的 GPT 模型，也称为 instructGPT，针对指令进行了微调。最新的 Chat Completion 则对应于 `gpt-3.5-turbo` 模型，即广为人知的 ChatGPT 模型。尽管 OpenAI 将 `text-davinci-003` 划归至 GPT3.5 系列，但 `gpt-3.5-turbo` 的性能与性价比被公认为最佳。如果你想了解各种模型之间的差异、 GPT-4 、 GPT 模型的发布论文或者更多 OpenAI 模型，可以参考 [Models](https://platform.openai.com/docs/models)。

> 还有一种 Code Completion ，对应于 GitHub Copilot 底层使用的 Codex 模型（同样基于 GPT3，但针对程序设计语言进行微调），但是 Codex 服务即将被取消，因此不再赘述。

使用 Completion API 非常直观且简单，强烈建议你去 [OpenAI Playground](https://platform.openai.com/playground?mode=complete) 亲自尝试。对于 Text Completion ，右边可能有一些参数，你暂时保持默认即可。在 Mode 中切换为 Chat 就可以调用 Chat Completion ，尽管它们之间存在一定差异，但实际上，Chat API 更为简便易用。

只需牢记模型的核心功能是进行猜测，而不是思考，就能顺利掌握并运用这两个 API。

### Prompt Engineering

在《西部世界》这部作品中，管理员通过与 AI 对话来进行编程和调试，这与大型语言模型的使用方式颇为相似。当使用 Completion API 时，由用户提供的文本被称为“prompt”，即给模型的提示语，以便让它根据用户的提示生成文本。编写提示语并不困难，只要 prompt 足够接近自然语言（即听起来像人话），对于模型来说就是有效的。然而，要编写能实现特定任务的提示语，并非易事。

Prompt Engineering 是一个复杂、庞大的主题，不是本文的关注焦点。如果你想了解更多，可以参考以下资料：

- [Text Completion 文档中的 Prompt design 介绍](https://platform.openai.com/docs/guides/completion/prompt-design)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [Prompt Engineering by Lilian Weng](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)

## 语言模型高级工程实践

### Embeddings API

GPT-4 相较于 3.5 在文本生成能力上并没有显著提升，然而，除了多模态之外，GPT-4 最令人惊叹的升级无疑是支持 8k 和 32k token 上限。这种提升让人不禁回想起过去几十年存储芯片的发展历程，从最初的千字节一直到如今的若干 PB 存储。我们可以预见，在不久的将来，人们可能会用同样的心情回顾如今的语言模型，从处理数千 token 到未来更高水平的演变。

然而，在某些应用场景中，仅仅依靠在 prompt 中传达信息来实现目标是不够的，尤其是考虑到成本和效果。例如，GitHub Copilot 无法将所有代码都写入 prompt，类似 Notion AI 这样的文档辅助工具也不可能将所有用户文档加载至 prompt。在这种情况下，如果想向模型传授大量知识，并希望它在推理过程中回忆这些知识，通常会使用 Embeddings API。

Embeddings API 可以计算文本间的关联度，一个 embedding 实际上是一个浮点数向量，如：

```
[
  -0.006929283495992422,
  -0.005336422007530928,
  ...
  -4.547132266452536e-05,
  -0.024047505110502243
]
```

Embeddings API 的主要功能是获取文本的 embedding。这些 embedding 可以看作提取自文本的特征，可用于分类、聚类和推荐。但更常见的用途是搜索。例如，在代码补全应用中，我们可以获取用户代码库的大量文本的 embedding。当需要补全代码时，我们先进行搜索，找到与当前上下文相似的应用场景，然后通过 prompt 将搜索结果告诉模型，从而实现“回忆长期记忆”的效果。

已有许多应用通过 Embeddings API 成功实现了长期记忆效果，例如 [researchGPT](https://github.com/mukulpatnaik/researchgpt)，[DocsGPT](https://github.com/arc53/docsgpt)，[Paul Graham GPT](https://github.com/mckaywrigley/paul-graham-gpt) 等，在 cookbook 中也有不少案例值得学习。

除此之外，还有一些专注于将使用 embedding 处理大量数据的过程抽象，如 [LlamaIndex (GPT Index)](https://github.com/jerryjliu/llama_index)，它的目标在于提供一个集中处理 embedding 的接口，将私有数据融入任何语言模型。LlamaIndex 将使用 embedding 的过程总结为*上下文学习模式*。

> 实际上，embedding 是 GPT 模型底层架构 Transformer 架构中的概念，以下是 GPT-4 所生成的对这个概念的介绍：

Transformer 架构是一种在自然语言处理（NLP）领域广泛应用的深度学习模型，它在机器翻译、文本生成、文本分类等任务中取得了显著的成果。在 Transformer 架构中，embedding 是一种关键技术，它可以将离散的词汇信息（如单词、字符等）转换为连续的向量表示，从而便于模型进行计算和处理。

具体而言，embedding 是一种将文本中的词汇映射到高维空间的向量表示方法。这些向量可以捕捉词汇间的语义和语法关系，使得相似或相关的词汇在向量空间中具有相近的位置。Transformer 模型通过学习大量的文本数据，形成了一个预训练的词汇嵌入矩阵，这个矩阵可用于将输入文本转换为对应的词嵌入向量。

OpenAI Embeddings API 是一个用于获取预训练词嵌入的工具，它可以直接调用 OpenAI 预训练好的 Transformer 模型，以便为用户提供高质量的词嵌入。使用 OpenAI Embeddings API 创建 embedding 的过程非常简单，用户只需将待转换的文本作为输入，API 会返回相应的词嵌入向量。这些向量可以用于各种 NLP 任务，如文本分类、情感分析、命名实体识别等。

以下是一个使用 OpenAI Embeddings API 调用模型创建 embedding 的简单示例：

```python
import openai

# 配置 API 密钥

openai.api_key = "your_api_key"

# 调用 Embeddings API

response = openai.Embedding.create(
model="your_preferred_model", # 如："gpt-3"
text="这是一个示例文本。"
)

# 获取词嵌入向量

embedding_vector = response["embedding"]
```

通过这种方式，用户可以轻松地利用 OpenAI 的强大预训练模型为自己的 NLP 任务生成高质量的词嵌入。

### Fine-tuning

Fine-tunning 是 Text Completion 和 Chat Completion 的不同之处。它提供基于 GPT 模型进行微调训练的能力，从而使模型更具独特性，且输出更为精准。事实上，ChatGPT 也是通过 fine-tuning 实现的。对于具有特殊输入或输出格式的应用，例如聊天输出需要符合对话格式且连贯，或者代码补全需要符合程序设计语言的语法规则等，fine-tuning 可以帮助模型更好地适应这些特殊场景。OpenAI 的 Fine-tuning 文档提供了相关[案例](https://platform.openai.com/docs/guides/fine-tuning/example-notebooks)，甚至可以通过 fine-tuning 完成分类、验证这样的任务。

尽管听起来完美，但 fine-tuning 需要准备大量的数据，且对数据的质量有比较高的要求，所以在实际应用中，fine-tuning 通常作为一种极限优化手段，通常我们在进行 fine-tuning 之前，通过实现 MVP （Minimum Viable Product）来验证其可行性。

目前最实用的模型 `gpt-3.5-turbo` 是 Chat Completion 模型，所以并不适用于 fine-tuning。但鉴于同为 Text Completion，即将发布的 GPT-4 API 很可能会支持 fine-tuning，目前已经有一些应用，将 GPT-4 用于精确性要求极高的任务，例如 [Lume](https://www.lume-ai.com/) 将 GPT-4 模型用于转换数据格式。

### 语言模型应用框架（WIP）

直接使用 OpenAI API 接入 GPT 模型是最简单也是最快的方式，但当人们反复重写同样功能的代码时，软件工程就会出手。目前市面上有两款语言模型应用框架，他们功能各有不同，但都提供了 prompt templating, chain, memory 等类型的功能：

- [Microsoft / Semantic Kernel](https://github.com/microsoft/semantic-kernel)
  微软绝对是接入 OpenAI 最快也是最早的大型公司，他们发布的“语义内核”是非常具有权威性的概念。
- [LangChain](https://github.com/hwchase17/langchain)
  而 LangChain 作为类似的开源项目，已经有[近千应用](https://github.com/hwchase17/langchain/network/dependents) 的用户群体。甚至连[微软的最新研究](https://github.com/microsoft/MM-REACT)，也用上了 LangChain，毫无疑问，LangChain 是目前最具有实际应用价值的语言模型应用框架之一。
- [NVIDIA Generative AI for Enterprises](https://www.nvidia.com/en-us/ai-data-science/generative-ai/)
  任何现代 AI 都离不开 NVIDIA 的计算芯片，他们同样在最近（2023-3-21）宣布了他们的 LLM 云服务，相比于上面两款开源的解决方案，NVIDIA 推出企业级的 SaaS 服务，如果预算丰富，使用它一定可以得到更好的效果。除了文本生成以外，NVIDIA 还提供蛋白质大模型服务、视觉内容、端到端大模型框架，具体内容可以参考[这篇文章 NVIDIA 的 LLM 介绍文章](https://www.nvidia.com/en-us/glossary/data-science/large-language-models/)、[NVIDIA NeMo Service 官网](https://www.nvidia.com/en-us/gpu-cloud/nemo-llm-service/)、[NVIDIA 生成式 AI 云服务发布文章](https://nvidianews.nvidia.com/news/nvidia-brings-generative-ai-to-worlds-enterprises-with-cloud-services-for-creating-large-language-and-visual-models)。在这一天他们还发布了 AI 工作站、AI 计算云服务，但是这些服务更像是为语言模型开发而非应用提供的。

### 开源大语言模型

OpenAI 在早期就开源了 [GPT-2](https://github.com/openai/gpt-2) 模型与 [Whisper](https://github.com/openai/whisper) 模型，后者用于语音识别。尽管如此，最近几年 OpenAI 经常由于不够 Open 而被人讽刺和诟病，而在大语言模型领域，真正 Open 的却是投资元宇宙亏钱的 Meta。从早期的 [OPT](https://ai.facebook.com/blog/democratizing-access-to-large-scale-language-models-with-opt-175b/) 模型，到现在的 [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)，Meta 一直没有吝啬自己的研究成果，于是他们的模型，也是开源大语言模型中最为广泛应用与传播的。

LLaMA 更像是早期的 GPT 模型，这类模型是大语言模型最原本的样子，它们的输出结果通常杂乱无章，对于人类来说实用价值第。但只要对这些模型进行一定程度的 fine tune，就可以实现像 GPT-3 (instructGPT) 或 ChatGPT 这样的效果。斯坦福大学就基于 LLaMA 模仿 GPT-3 针对 指令（Instruction）的 fine tune 过程，发布了 [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)。此外，清华大学唐杰教授的团队，一直在进行语言模型的研究，它们的 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) 效果惊人，由于 Meta LLaMA 的训练数据中不包含简体中文，对于中文开源大模型也只有 GLM 模型具有先进的效果。

在社区高手的发酵下，诞生了一系列方便部署的开源项目，接下来在此对这些项目进行总结和介绍。

- [ggerganov / llama.cpp](https://github.com/ggerganov/llama.cpp)
  作者 Georgi Gerganov 实现的推理方法，可以把 LLaMA、Alpaca 高效地运行在 CPU 上，甚至有人用手机运行了 LLaMA 模型。
- [THUDM / ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
  中文开源大语言模型的唯一选择，由清华大学唐杰教授团队开发，效果惊人。
- [tloen / alpaca-lora](https://github.com/tloen/alpaca-lora)
  使用 [LoRA](https://github.com/microsoft/LoRA) 技术同样以 instruct 风格 fine tune 的 Alpaca 模型，有一款 [UI](https://github.com/lxe/simple-llama-finetuner) 可以更方便的使用这个仓库。
- [nichtdax / awesome-totally-open-chatgpt](https://github.com/nichtdax/awesome-totally-open-chatgpt)
  一个介绍开源 ChatGPT 替代品的文档仓库，对开源仓库的范围进行了合理的总结和分类，包括了上面介绍的三款模型模型。如果你想要部署一个 ChatGPT 的替代品，可以从这里开始。
- [LianjiaTech / BELLE](https://github.com/LianjiaTech/BELLE)
  链家 / 贝壳找房开源的基于 alpaca 的中文模型。
- [Const-me / Whisper](https://github.com/Const-me/Whisper)
  同样基于 Georgi Gerganov 的方法，推理 OpenAI Whisper 模型。
- [bigscience-workshop / petals](https://github.com/bigscience-workshop/petals)
  BigScience 提供了一种像 BT 下载一样合作运行大语言模型的方法，可以让家用电脑也能运行 100B+ 参数的模型。由于在自建大模型这个场景下，推理实际上对计算资源的占用率并不高，这样的方法可以提高计算资源的利用率，如果经过良好设计，实际上是一种非常高效的方法。BigScience 声明这样调用可以比将任务外包给云端完成的方式快 10 倍，并且可以对模型进行 fine tune。
- [Alpaca-7B Truss](https://github.com/basetenlabs/alpaca-7b-truss)
  Alpaca-7B 的 Truss 实现，Truss 是一个用于开发和部署机器学习模型的开源模型服务框架，根据他们提供的[在线 demo](https://chatllama.baseten.co/)，效果很棒。

如果你有更好的开源大语言模型，欢迎通过邮件与我分享。

## 不远的未来

### AI 带来的产业升级与下一发银弹

根据 [ChatPDF](https://www.chatpdf.com) 对 OpenAI 论文[大语言模型对劳动力市场的潜在影响力](https://arxiv.org/abs/2303.10130) 这篇论文的分析：

> 约 80％的美国劳动力可能会受到 GPT 模型引入的影响，其中约 19％的工人可能会看到至少 50％的工作任务受到影响。这种影响不仅限于高生产力增长行业，而且涵盖了所有薪资水平。根据第 35 页的 ResumeBuilder.com 的数据，有四分之一的公司已经使用 ChatGPT 替换了工人。

有一些人担忧大语言模型可能会影响人类的薪资，甚至可能会替代人类，在艺术绘画这个领域内，这种担忧非常广泛。从另一个角度来讲，艺术家们似乎抗拒与 AI 合作，没有画师积极参与 AI 开发，这显然不是一个健康的发展过程。

> 尽管这样的担忧显然是有理由的，[打鱼记·上（Midjourney 漫画）](https://mp.weixin.qq.com/s/Co4c41JXSvxcQbeJjIjLgQ)

假设，AI 真的为人类带来了产业升级，也许我们可以幻想在不远的将来，领域专家们将像参与到软件开发一样参与到 AI 的研发中，产生高质量的数据用来对 AI 进行调参。未来的 AI 工程就像当今的软件工程一样繁荣发达，专家们研发出可以让普通开发者简单快速上手的 AI 框架，甚至可以让行外的普通人花短时间培训就可以进入 AI 行业，到这个时候，AI 专业将成为继土木工程、计算机科学之后的下一个银弹。

### AI 与软件工程师

GitHub Copilot 是生成式语言模型 AI 对软件工程师们的第一次重击，它已经服务了开发者们一年多的时间。最近，GitHub 又推出了更强大的 [Copilot X](https://githubnext.com/projects/copilot-for-docs)

> Copilot 这个词在微软内部似乎已经成为了生成式大语言模型应用的代名词。

已经有许多专注于开发软件工具的 AI 初创企业纷纷涌现。它们包括：

- [Watermelon](https://www.watermelontools.com/)
  通过高亮显示代码块并利用 Git、通讯系统和工单系统找到重点信息
- [codeium](https://codeium.com/)
  Codeium 是一款将人工智能整合至编程过程的高效开发工具，支持 40 多种编程语言和 20 多个编辑器，通过智能搜索、代码生成等 AI 功能，助力开发者与团队提高编码效率和产品交付速度。
- [bloop](https://bloop.ai/)
  利用 GPT-4 智能理解代码库，并实现高效语义代码搜索
- [zapier](https://zapier.com/)
  源自 [Hacker News](https://news.ycombinator.com/item?id=4138415) 的初创公司，率先[将语义编程引入](https://news.ycombinator.com/item?id=35263542)了 IFTTT 系统中。

欢迎补充。

AI 显然会改变软件工程师的工作方式，但距离替代软件工程师，还远远不够，在文章[为什么 AI 无法取代软件工程师](https://softwarecomplexity.com/why-ai-wont-replace-software-engineers)中，作者认为，

> GPT-4 根据前面提到的文章生成的总结：
>
> 虽然 AI 工具如 co-pilot 能提高软件工程效率，但软件工程师仍不可或缺。他们不仅负责编写代码，还需理解复杂系统、管理复杂性、调试代码及维护任务。软件工程师在编写代码、维护质量与交付功能之间寻求平衡，确保系统质量。因此，尽管 AI 技术在某程度上辅助工程师，但优秀的软件工程师和领导者始终是关键。

这篇文章在 [Hacker News](https://news.ycombinator.com/item?id=35247239) 上引起了共鸣和讨论，下面将提供 AIGC 总结，但我强烈建议阅读原文，HN 上的大胡子们提供了很多有价值的观点。

> GPT-4 根据 Hacker News 中的讨论生成的总结：
>
> 尽管人工智能工具如 ChatGPT 在提高编码效率方面具有潜力，但它无法完全取代软件工程师。ChatGPT 的代码不总是正确且有时不符合要求，且通常调用已知解决方案而非创新方案。虽然它在提供代码模板、帮助文档和搜索方面有所帮助，但在实际编写和部署生产代码方面仍有待完善。人们对人工智能在编码中的作用看法不一，需注意 AI 可能引入的复杂性。总之，软件工程师在分析软件复杂性方面仍是不可或缺的。

### AI 与教育

对许多人来说，他们可能并未意识到 AI 在他们的教育生涯中扮演了重要角色。事实上，搜索引擎就是现代 AI 的一个典型应用，而绝大多数大学生在学习过程中都使用过搜索引擎。除此之外，还有其他 AI 技术，例如机器翻译和语音合成，在我的英语学习中发挥了巨大作用。自从 Google 还在中国市场时，我就开始利用 Google 翻译来翻译英文电脑游戏和英文游戏网站，这些经历对我英语的提高帮助非常大。

我想强调的是，AI 技术通过各种互联网应用已经渗透到我们的日常生活中。因此，我们不应该因为大型语言模型的出现而忽略 AI 在过去几年的发展。

### 伦理、法律、道德、监管与安全

从 AI 这个概念诞生以来，从未像今天一样引起了人们的广泛关注和讨论。AI 不仅是第四次产业革命的核心，也是推动社会进步和变革的重要力量。但是，AI 也给伦理道德规范和社会治理带来了挑战，比如数据隐私、算法歧视、责任归属、人机关系等问题。因此，建立和完善 AI 的伦理规范和治理机制，保障 AI 的安全可控和可信赖，是当务之急。

> 以上内容由 Bing AI 生成

这些都是比较庞大且复杂的话题，如果以后有机会，我会单独写一篇文章来讨论这些问题，目前，我推荐阅读以下资料：

- [OpenAI CEO warns that GPT-4 could be misused for nefarious purposes](https://www.theregister.com/2023/03/20/openai_warns_that_gpt4_could/)
- [Google Search's guidance about AI-generated content](https://developers.google.com/search/blog/2023/02/google-search-and-ai-content)
- [Nearly Half of Firms Are Drafting Policies on ChatGPT Use](https://www.bloomberg.com/news/articles/2023-03-20/using-chatgpt-at-work-nearly-half-of-firms-are-drafting-policies-on-its-use)
- [Language models might be able to self-correct biases—if you ask them](https://www.technologyreview.com/2023/03/20/1070067/language-models-may-be-able-to-self-correct-biases-if-you-ask-them-to/)
- [GPT-4 论文竟有隐藏线索：GPT-5 或完成训练、OpenAI 两年内接近 AGI](https://www.qbitai.com/2023/03/42885.html)
- [The End of Front-End Development](https://www.joshwcomeau.com/blog/the-end-of-frontend-development/)
- [Mozilla.ai announcement.](https://blog.mozilla.org/mozilla/introducing-mozilla-ai-investing-in-trustworthy-ai/)
- [These new tools let you see for yourself how biased AI image models are](https://www.technologyreview.com/2023/03/22/1070167/these-news-tool-let-you-see-for-yourself-how-biased-ai-image-models-are/)

#### 隐私数据训练与联邦学习

联邦学习可以解决许多 ML 项目面临的数据保护和隐私问题，比如数据敏感性、组织隔离、用户隐私等。[Flower](https://flower.dev/) 是一个开源框架，用于在分布式数据上训练 AI 模型，其使用联邦学习方法，将模型移动到数据而不是将数据移动到模型，以实现监管合规性（例如 HIPAA）和其他无法实现的 ML 用例。Flower 使你可以在许多用户设备或“数据孤立”（独立的数据源）中分布的数据上训练 ML 模型，而不必移动数据，这种方法称为联邦学习。Flower 支持 PyTorch、TensorFlow、JAX、Hugging Face、Fastai、Weights＆Biases 等 ML 项目中的所有其他工具，适用于个人工作站、Google Colab、计算集群、公共云实例或私人硬件等各种环境。此外，Flower 提供一个教程，展示了如何使用该框架。

联邦学习还可以解决很多 ML 项目中的问题，如：

- 生成 AI：很多情况需要敏感数据，用户或组织不愿意上传到云端，但使用联邦学习可以从个人设备中使用敏感数据，同时保护用户隐私。
- 医疗保健：可以比医生更好地训练癌症检测模型，但没有任何单个组织拥有足够的数据。
- 金融：个人银行面临数据法规的约束，无法培训良好的模型，因此金融欺诈防范变得更加困难。
- 自动驾驶：单个汽车制造商无法收集涵盖所有边缘情况的数据。
- 个人计算：用户不想让某些数据存储在云中，因此使用联邦方法开启了从个人设备使用敏感数据的大门。

Flower 是一种优秀的 AI 技术，其基于联邦学习方法，能够训练分布式且敏感的数据，使得模型不必移动数据，解决了许多 ML 项目面临的数据保护和隐私问题。

> _以上内容由 ChatGPT 生成_
