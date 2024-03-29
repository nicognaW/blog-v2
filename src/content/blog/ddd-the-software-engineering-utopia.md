---
author: Nico
pubDatetime: 2024-01-24T02:18:00.000Z
title: 暴论：DDD （领域驱动设计）是软件工程乌托邦
postSlug: ddd-the-software-engineering-utopia
featured: true
draft: false
tags:
  - SE
  - Architecture
ogImage: ""
description: ""
---

## 暴论

大概在两年前，当时还在写 Java ，感觉 Spring Boot 写起来很丑陋，并且代码组织起来怎么看怎么不舒服，于是一口咬定一定是架构的问题。总之就开始满世界寻找最“先进”和“现代”的架构是什么，对于架构来说毫不意外答案会是微服务架构，而微服务是一个技术尺度的架构，对于设计软件帮助不大。
而与微服务经常并列出现的，就是 DDD ，领域驱动。

不得不说领域驱动这个名字看着就很酷，而它的概念更酷，年幼而缺乏经验的我，当初很快就把 DDD 当成了所有问题的唯一解，开始尝试、实践，并告诉所有人你们的架构太 low 了，快换成 DDD 吧。然而在经历过一次失败，以及对 DDD 原书以及两本名家批注进行反复阅读和研究之后，我得出了一个完全相反的结论 —— DDD 是乌托邦。

![](https://raw.githubusercontent.com/AppFlowy-IO/docs/main/uml/output/DDDLayeredArchitecture.svg)

上图是一张来自于 [appflowy 文档](https://docs.appflowy.io/docs/documentation/software-contributions/architecture/domain-driven-design)的 DDD 分层架构图，虽然这是一张 DDD 的架构图，但它也符合在软件行业自古以来就有的三层架构。

![](https://upload.wikimedia.org/wikipedia/commons/5/51/Overview_of_a_three-tier_application_vectorVersion.svg)

这一点是非常巧妙的，DDD 实际上和三层架构并不冲突，而是扩展指导了第二层的实现方式，也就是针对原本的三层架构，做了更多的约束和要求。
用 DDD 自己的语言讲就是缩小了解空间，而解空间越小，问题空间也就越小，所以与大多数软件都适用三层架构不同，更少的软件将适用 DDD 。

那么具体哪些问题适用使用 DDD 作为解呢？其实原书的标题说的很明白 —— 《领域驱动设计：软件核心复杂性应对之道》，就是“软件核心复杂性”，实际上讲业务复杂性更顺口一些。
根据 DDD ，只有极高的业务复杂性将会适用 DDD ，那具体多高才用 DDD 呢？这是一个留给后世思考的问题，而我的答案是，**不论多高，都不用 DDD** 。

乌托邦的意思是完美的社会，而 DDD 描述的是一个完美的软件架构，如果讲环境理想化，我们有无穷无尽的人力，每位工程师直接的配合都是完美的，对业务、项目、技术的理解也都完美并且一致，那么不论任何软件都应该用 DDD 。
但现实的平面却无比粗糙，我们通常有不够用的人力，糟糕的团队合作，工程师的水平参差不齐，沟通几乎充斥着误解和矛盾，所以我认为不论软件多复杂，都不应该用 DDD。所以哪怕当软件复杂度高到 DDD 开始 make sense 时，就算使用 DDD ，也无济于事。

那如果 DDD 不是复杂软件的解，解是谁呢。我不知道，就像我不知道人类社会的解一样。

## 分层架构与 MVC

刚学 Spring Boot 时，每写一个 HTTP API ，都要写一个 `@Controller` annotated component ，以及一个 `@Service` 和一个 Repository （忘了用什么注解了）。这个东西在国内经常被称为“ MVC ”。而 MVC 的全称是 Model–view–controller ，和 Service 以及 Repository 属于是毫无关联。

MVC 其实并没有要求写 Service 和 Repository ，它只是要你在 `Controller` 里面返回 `ModelAndView` ，至于你的 Model 是数据库连接层还是一个变量，它不管的。我们通常写的这三层，其实是在函数尺度实现上文提到的三层架构。
如果我写一个 `Controller` 函数，然后直接原地返回一个字符串变量，那也可以算 MVC ，甚至可以算三层架构。

## 分层架构带来的可测试性

在写完一个接口时，想测试一下写的行不行，会怎么做呢。在刚学 Web 开发时，很多人会本地开一个数据库服务器，然后打开 API 测试工具，输入 localhost ，然后填好参数发一个请求。这一套流程在行话上叫做端到端测试，并且是手动的端到端测试。

在软件工程质量保证届，我还有一个暴论，就是“源代码质量=测试覆盖率”，这个地方的测试覆盖率，并不是说你手动多测几个情况就行了，而是要使用自动测试，并且尽可能大量编写单元测试。

单元测试中的单元，最常见的就是一个函数，而我编写的程序是一个 Web 服务，展开来讲是一个处理 HTTP 请求，然后执行一段业务代码，最后调用数据库 API 的流程。
如果不进行合理的分层，我们就只能本能地做手动端到端测试，每一次执行测试必须由一个 HTTP 请求开始，最后以一次完整的数据库操作结束。

在这个流程里如果我们遵守分层架构，就会产生三个函数，如果对这三个函数进行合理的解耦，就可以针对不同的阶段编写多个单元测试了。
