---
title: C代码随记
date: 2022-09-19 10:24:43
tags:
categories:
- C++ & C
---

##### 1、static_cast C++强制类型转换操作符 

是一种显示转换，就是告诉编译器，这种损失精度是在已知的情况下进行的；例如下面的a=b则为隐式转换；当编译器执行隐式转换时，会有一个warning，提示可能存在精度损失。<!--more-->

```c++
double a = 1.99;
int b = static_cast<double>(a); //相当于a = b；
```

使用static_cast 可以找回存放void* 指针中的值

```c++
double a = 1.99;
void * vptr = &a;
double * dptr = static_cast<double*> vptr;
std::cout<<*dptr<<std::endl;
```

