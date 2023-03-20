---
author: Nico
pubDatetime: 2022-09-28T10:08:32.000Z
title: Victor Mono——最酷的编程字体！原理与配置！
postSlug: cool-programmer-font
featured: true
draft: false
tags:
  - Cool stuff
  - Tools
ogImage: ""
description: 本文介绍了一款超酷的编程字体——Victor Mono，它能把代码中的关键字和注释变成手写体，看起来非常酷炫！并介绍了在 Jetbrains 和 Visual Studio Code 中的配置方法。
---

# Victor Mono——最酷的编程字体！原理与配置！

## [Victor Mono](https://rubjo.github.io/victor-mono/)

![117447088-53e03300-af3d-11eb-84e2-df1713e77019[1].png](https://cdn.hashnode.com/res/hashnode/image/upload/v1664295042981/5qYHngIWc.png align="left")

[Victor Mono](https://rubjo.github.io/victor-mono/) 是一款超酷的编程字体，配置这款字体后，代码里的关键字、注释有一些会变成**_手写体_**，看起来非常的酷！

## [Cascadia Code](https://github.com/microsoft/cascadia-code)

其实可以做到这样效果的字体还有很多，比如微软的 [Cascadia Code](https://github.com/microsoft/cascadia-code)，这款字体是 Windows 11 内置的，可以默认设置在 Windows Terminal 内，但是 Windows Terminal 并不负责实现把部分文字变成手写体。

## 原理

实际上这些变成手写体的字，并不是真的把他们换成了另一个手写字体，本质上是把他们变成了 `italic` 样式，而字体中所有的 `italic` 字体，都是手写体。

![image.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1664295401609/h6MpvMV3U.png align="left")

## 安装与配置

安装 [FiraCodeiScript](https://github.com/kencrocken/FiraCodeiScript)、 [Cascadia Code](https://github.com/microsoft/cascadia-code) 或 [Victor Mono](https://rubjo.github.io/victor-mono/) 任意一款字体，都可以实现这样的效果，后文以 **Victor Mono** 为例。

### Jetbrains

对于 Intellij Idea 以及其他 Jetbrains IDE，首先在 `Editor > Font` 内选择字体

![image.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1664295567627/MY9fn-7Q2.png align="left")

之后在配置 `Editor > Color Scheme` 就可以啦，这个 Color Scheme 配置项简单易懂，相信你玩一玩就可以上手！

![image.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1664295645787/N6hOgD60E.png align="left")

### Visual Studio Code

Code 的配置方法要稍微复杂一点，你可以参考[这篇文章](https://www.stefanjudis.com/blog/how-to-enable-beautiful-cursive-fonts-in-your-vs-code-theme/)（可直接抄他作业）。

### 其他编辑器 / 工具

相信你已经发现了其中的模式，在代码编辑器里，规定哪些代码用什么样的样式渲染的选项通常叫 `Color Scheme`，所以对于任何支持这一特性的编辑器，你只要[这样搜索](https://www.google.com/search?q=vim+color+scheme)，就能得到答案！~~来吧，试试看！~~

另外，如果你想看看别的酷编程字体，你可以看看[这个网站](https://www.programmingfonts.org/)。
