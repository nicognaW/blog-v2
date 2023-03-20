---
author: Nico
pubDatetime: 2023-03-20T00:00:00.000Z
title: 语言模型应用的高级工程实践
postSlug: advanced-engineering-practices-of-language-model-applications
featured: false
draft: true
tags:
  - AI
  - LLM
ogImage: ""
description: 简单科普 OpenAI Platform，并介绍一些高级工程实践和有价值的开源大模型。
---

# 语言模型的高级工程实践

自从 OpenAI 发布了 ChatGPT ，并逐步开放了 `text-davinci-003` 、 `gpt-3.5-turbo` 等模型。大语言模型的工程应用出现了百家争鸣的现象。本文首先简单介绍 OpenAI API 的基础使用方法，然后介绍几个在本文编写时，比较前沿的一些高级工程实践。此外，还会给出一些参考案例，以及当下开源大语言模型的概览。

> _本文会大面积使用大语言模型生成的文本。_

## OpenAI API

### ChatGPT 与 OpenAI API 的关系

ChatGPT 是 OpenAI 使用 GPT 系列模型针对对话场景进行调参（fine tue）研发出的一款产品。由于爆炸式出名，也将它底层使用的 `gpt-3.5-turbo` 模型称为 ChatGPT 模型，也有人将使用这个模型的 Chat Completion API 称为 ChatGPT API。
OpenAI API 与 ChatGPT 这样的 c 端产品不同，它是一系列供开发人员使用的 API。

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

### Completion API，语言大模型魔法

Completion API 是语言模型提供的最直观的能力之一，也是实际应用场景中最常用的 API。

它可以阅读你提供的文本，然后猜测可能出现在这个文本后面的文本。这个猜测的标准来自于它读过的上百亿文本。与直觉相反，语言模型并不会明确地进行逻辑思考，而是通过猜测，写出最像样子的文本。

虽然是靠硬猜实现，GPT 模型的效果令人惊叹，是因为其强大的训练数据规模，GPT3 模型的训练数据集的文本大小是 570 GB，本文截至写完，文件的大小是 3 kb，也就是说 GPT 读过的文本量相当于 20 亿篇本文长度的文章。

> 实际上 GPT3 模型早在 2020 年 5 月就发布了，我也曾经体验过一些早期的 GPT3 应用，有一些也是 chat bot 这样的形式，但与 ChatGPT 的效果相差甚远。主要是因为 OpenAI 在 GPT3 模型的基础上，加了[人类反馈的强化学习](https://openai.com/blog/chatgpt#methods)，来微调模型，使其迅速适应对话场景。

Completion API 有三种，分别是 Text Completion, Code Completion 和 Chat Completion ，其中 Text 对应的是更老的 GPT 模型，也被称为 instructGPT，针对指令进行了微调，Code 对应的是 GitHub Copilot 底层使用的 Codex 模型（同样是 GPT3 模型，但为程序设计语言微调）。而最新的 Chat Completion 对应的是 `gpt-3.5-turbo` 模型也就是广为人知的 ChatGPT 模型。虽然现在 OpenAI 把 `text-davinci-003` 分到了 GPT3.5 系列，但 `gpt-turbo-3.5` 的性能和性价比是公认最好的。如果你想了解各种模型之间的差异、 GPT-4 、 GPT 模型的发布论文或者更多 OpenAI 模型，可以参考 [Models](https://platform.openai.com/docs/models)。

Completion API 的使用都比较简单并且符合直觉，强烈建议你去 [OpenAI Playground](https://platform.openai.com/playground?mode=complete) 进行一个体验。对于 Text Completion ，右边可能有一些参数，你暂时保持默认即可。在 Mode 中切换为 Chat 就可以调用 Chat Completion ，虽然有一些不同，但是 Chat 实际上更加简单。

只要你记住，模型做的事情是猜测，你就可以无障碍得理解和使用这两个 API。

### Prompt Engineering，西部世界编程

在西部世界这部片中，管理用通过和 AI 对话来进行编程，这和大语言模型的使用方式非常相似。你在使用 Completion API 时，由你来提供的文本叫做 prompt ，也就是给模型的提示语，让它用你的提示语来生成文本。编写提示语并不难，只要你的提示语像人话，对于模型来说就是有效的，但是想要编写出能实现特定工作的提示语，并不简单。

Prompt Engineering 是一个复杂的话题，并且不是本文的重点，如果你想了解更多，可以参考以下资料：

- [Text Completion 文档中的 Prompt design 介绍](https://platform.openai.com/docs/guides/completion/prompt-design)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)

### Embeddings API，为模型提供长期记忆
