---
title: VS code环境设置
date: 2022-08-29 16:07:57
tags:
categories:
- 工具
---

### 给VS code 配置临时环境 

经常用vs code开放python项目，如果调试远程服务器的项目时，常用做法是在终端配置环境变量，然后在终端运行程序，通过 pdb 库调试，查看中间结果，个人觉得很不方便，本文介绍一下通过launch.json配置环境，可以进行debug调试

<!--more-->

"env"中的设置就是一些环境变量，或者局部变量 :leaves:

"cwd"设置当前工作环路径为

```json
{
    "veersion": "0.2.0",
    "configurations":[
        {
            "name": "Python:Current File",
            "type": "python",
            "request": "launch",
            "program"： "${file}",
            "console": "integratedTerminal",
            "env": {
                "LD_LIBRARY_PATH": "xxxxxxxxxxxxxx"
                "MXNET_CUDNN_AUTOTUNE_DEFAULT": "0"
            },
        	"cwd":"${fileDirname}"
        }
    ]
}
```

 在项目下有launch.json文件，在env中设置我们需要的环境变量，

可以在.vscode文件夹下创建settings.json文件，可配置debug时使用的python 环境，设置内容如下：

```json
{
    "python.pythonPath": "/data/ymliu29/5_anaconda2s/anaconda2/bin/python"
}
```

