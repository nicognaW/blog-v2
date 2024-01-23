---
author: Nico
pubDatetime: 2023-07-31T15:14:53.216Z
title: AWS 与无尽地狱
postSlug: aws-sucks
featured: false
draft: false
tags:
  - Cloud
  - AWS
ogImage: ""
description: 8 说了，看吧
---

> 终于，AWS EKS在夏日黄昏唤醒，柳暗花明般欣慰；
>
> AWS Load Balancer，麻烦连连，似穿越繁复丛林的迷途。
> 
> 空荡的办公室，澎湃的音乐在角落起舞，洒下坚定的光芒；
> 
> 犹如万籁俱寂后的第一缕晨曦，带来未知的新开始。
> 
> 劳动之余，面对窗外炽热的夏日，内心的欢喜如微风中的清凉。

现在时间大约是2023年7月31日的15:02，我正在尝试将一个应用部署到EKS。

目前，我正在创建IAM角色，参考的是["为Kubernetes配置Amazon VPC CNI插件，使用IAM角色为服务账户"](https://docs.aws.amazon.com/eks/latest/userguide/cni-iam-role.html)。

你猜怎么着，这个3步指南是前提条件中第三步的第一步，来自于["与Amazon EKS的Kubernetes Amazon VPC CNI插件一起使用"](https://docs.aws.amazon.com/eks/latest/userguide/managing-vpc-cni.html)。

我为什么需要这个呢？因为这是["安装Kubernetes服务账户"集群附加组件](https://docs.aws.amazon.com/eks/latest/userguide/service-accounts.html#boundserviceaccounttoken-validated-add-on-versions)
的第一步。

这看起来与我想要做的事情无关，对吧？那是因为这是前提条件的第三步，来自["安装AWS负载均衡控制器附加组件"](https://docs.aws.amazon.com/eks/latest/userguide/aws-load-balancer-controller.html)
，这又是[在EKS中使用L4负载均衡器的推荐方式](https://docs.aws.amazon.com/eks/latest/userguide/network-load-balancing.html)
的前提条件。

这里有个有趣的部分，我让GPT-4根据我的经验生成一个JSON：

```json
{
  "name": "将PI应用部署到EKS",
  "timestamp": "2023-07-31T15:02:00Z",
  "steps": [
    {
      "name": "在EKS中使用L4负载均衡器",
      "url": "https://docs.aws.amazon.com/eks/latest/userguide/network-load-balancing.html",
      "prerequisites": [
        {
          "name": "安装AWS负载均衡控制器附加组件",
          "url": "https://docs.aws.amazon.com/eks/latest/userguide/aws-load-balancer-controller.html",
          "prerequisites": [
            {
              "name": "安装Kubernetes服务账户",
              "url": "https://docs.aws.amazon.com/eks/latest/userguide/service-accounts.html#boundserviceaccounttoken-validated-add-on-versions",
              "prerequisites": [
                {
                  "name": "与Amazon EKS的Kubernetes Amazon VPC CNI插件一起使用",
                  "url": "https://docs.aws.amazon.com/eks/latest/userguide/managing-vpc-cni.html",
                  "prerequisites": [
                    {
                      "name": "为Kubernetes配置Amazon VPC CNI插件，使用IAM角色为服务账户",
                      "url": "https://docs.aws.amazon.com/eks/latest/userguide/cni-iam-role.html",
                      "steps": [
                        {
                          "name": "[...]",
                          "url": "[...]"
                        },
                        {
                          "name": "[...]",
                          "url": "[...]"
                        },
                        {
                          "name": "[...]",
                          "url": "[...]"
                        }
                      ]
                    }
                  ]
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

只是为了部署一个简单的应用到EKS，就需要一个6层嵌套的JSON，看看这有多离谱。
