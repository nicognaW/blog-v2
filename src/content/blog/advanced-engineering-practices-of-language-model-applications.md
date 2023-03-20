---
author: Nico
pubDatetime: 2023-03-20T00:00:00.000Z
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

# 语言模型的高级工程实践

自 OpenAI 发布 ChatGPT 以来，随着 text-davinci-003、gpt-3.5-turbo 等先进模型的逐步推出，大语言模型在工程应用领域呈现出百花齐放的景象。本文将首先简洁明了地介绍 OpenAI API 的基本使用方法，随后重点探讨在撰写本文时期，一些颇具前瞻性的高级工程实践。此外，还将提供若干实用的参考案例，同时概述当下开源大语言模型的发展状况。

> _本文含有使用大语言模型生成的文本。_

## OpenAI API 简介

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

Completion API 分为三类：Text Completion、Code Completion 和 Chat Completion。其中，Text Completion 对应较早期的 GPT 模型，也称为 instructGPT，针对指令进行了微调。Code Completion 则对应于 GitHub Copilot 底层使用的 Codex 模型（同样基于 GPT3，但针对程序设计语言进行微调）。最新的 Chat Completion 则对应于 `gpt-3.5-turbo` 模型，即广为人知的 ChatGPT 模型。尽管 OpenAI 将 `text-davinci-003` 划归至 GPT3.5 系列，但 `gpt-3.5-turbo` 的性能与性价比被公认为最佳。如果你想了解各种模型之间的差异、 GPT-4 、 GPT 模型的发布论文或者更多 OpenAI 模型，可以参考 [Models](https://platform.openai.com/docs/models)。

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
- [LangChain](https://github.com/hwchase17/langchain)

微软绝对是接入 OpenAI 最快也是最早的大型公司，他们发布的“语义内核”是非常具有权威性的概念。而 LangChain 作为类似的开源项目，已经有[近千应用](https://github.com/hwchase17/langchain/network/dependents) 的用户群体。

### 开源大语言模型

OpenAI 在早期就开源了 [GPT-2](https://github.com/openai/gpt-2) 模型与 [Whisper](https://github.com/openai/whisper) 模型，后者用于语音识别。尽管如此，最近几年 OpenAI 经常由于不够 Open 而被人讽刺和诟病，而在大语言模型领域，真正 Open 的却是投资元宇宙亏钱的 Meta。从早期的 [OPT](https://ai.facebook.com/blog/democratizing-access-to-large-scale-language-models-with-opt-175b/) 模型，到现在的 [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)，Meta 一直没有吝啬自己的研究成果，于是他们的模型，也是开源大语言模型中最为广泛应用与传播的。

LLaMA 更像是早期的 GPT 模型，这类模型是大语言模型最原本的样子，它们的输出结果通常杂乱无章，对于人类来说实用价值第。但只要对这些模型进行一定程度的 fine tune，就可以实现像 GPT-3 (instructGPT) 或 ChatGPT 这样的效果。斯坦福大学就基于 LLaMA 模仿 GPT-3 针对 指令（Instruction）的 fine tune 过程，发布了 [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)。此外，清华大学唐杰教授的团队，一直在进行语言模型的研究，它们的 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) 效果惊人，由于 Meta LLaMA 的训练数据中不包含简体中文，对于中文开源大模型也只有 GLM 模型具有先进的效果。

在社区高手的发酵下，诞生了一系列方便部署的开源项目，接下来在此对这些项目进行总结和介绍。

- [ggerganov / llama.cpp](https://github.com/ggerganov/llama.cpp)
  作者 Georgi Gerganov 实现的推理方法，可以把 LLaMA、Alpaca 高效地运行在 CPU 上，甚至有人用手机运行了 LLaMA 模型。
- [THUDM / ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
  中文开源大语言模型的唯一选择，由清华大学唐杰教授团队开发，效果惊人。
- [Const-me / Whisper](https://github.com/Const-me/Whisper)
  同样基于 Georgi Gerganov 的方法，推理 OpenAI Whisper 模型。
- [tloen / alpaca-lora](https://github.com/tloen/alpaca-lora)
  使用 [LoRA](https://github.com/microsoft/LoRA) 技术同样以 instruct 风格 fine tune 的 Alpaca 模型。

如果你有更好的开源大语言模型，欢迎通过邮件与我分享。

## 不远的未来

### AI 带来的产业升级与下一发银弹

根据 [ChatPDF](https://www.chatpdf.com) 对 OpenAI 论文[大语言模型对劳动力市场的潜在影响力](https://arxiv.org/abs/2303.10130) 这篇论文的分析：

> 约 80％的美国劳动力可能会受到 GPT 模型引入的影响，其中约 19％的工人可能会看到至少 50％的工作任务受到影响。这种影响不仅限于高生产力增长行业，而且涵盖了所有薪资水平。根据第 35 页的 ResumeBuilder.com 的数据，有四分之一的公司已经使用 ChatGPT 替换了工人。

有一些人担忧大语言模型可能会影响人类的薪资，甚至可能会替代人类，在艺术绘画这个领域内，这种担忧非常广泛。从另一个角度来讲，艺术家们似乎抗拒与 AI 合作，没有画师积极参与 AI 开发，这显然不是一个健康的发展过程。

> 尽管这样的担忧显然是有理由的，[打鱼记·上（Midjourney 漫画）](https://mp.weixin.qq.com/s/Co4c41JXSvxcQbeJjIjLgQ)

假设，AI 真的为人类带来了产业升级，也许我们可以幻想在不远的将来，领域专家们将像参与到软件开发一样参与到 AI 的研发中，产生高质量的数据用来对 AI 进行调参。未来的 AI 工程就像当今的软件工程一样繁荣发达，专家们研发出可以让普通开发者简单快速上手的 AI 框架，甚至可以让行外的普通人花短时间培训就可以进入 AI 行业，到这个时候，AI 专业将成为继土木工程、计算机科学之后的下一个银弹。

### 伦理、法律、道德、监管与安全

从 AI 这个概念诞生以来，从未像今天一样引起了人们的广泛关注和讨论。AI 不仅是第四次产业革命的核心，也是推动社会进步和变革的重要力量。但是，AI 也给伦理道德规范和社会治理带来了挑战，比如数据隐私、算法歧视、责任归属、人机关系等问题。因此，建立和完善 AI 的伦理规范和治理机制，保障 AI 的安全可控和可信赖，是当务之急。

> 以上内容由 Bing AI 生成

这些都是比较庞大且复杂的话题，如果以后有机会，我会单独写一篇文章来讨论这些问题，目前，我推荐阅读以下资料：

- [OpenAI CEO warns that GPT-4 could be misused for nefarious purposes](https://www.theregister.com/2023/03/20/openai_warns_that_gpt4_could/)
- [Nearly Half of Firms Are Drafting Policies on ChatGPT Use](https://www.bloomberg.com/news/articles/2023-03-20/using-chatgpt-at-work-nearly-half-of-firms-are-drafting-policies-on-its-use)
- [Language models might be able to self-correct biases—if you ask them](https://www.technologyreview.com/2023/03/20/1070067/language-models-may-be-able-to-self-correct-biases-if-you-ask-them-to/)
- [GPT-4 论文竟有隐藏线索：GPT-5 或完成训练、OpenAI 两年内接近 AGI](https://www.qbitai.com/2023/03/42885.html)
- [The End of Front-End Development](https://www.joshwcomeau.com/blog/the-end-of-frontend-development/)
