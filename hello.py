#!/usr/bin/env python3

print("Hello, World!")

# 添加一些额外的功能
def greet(name):
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(greet("World"))
    
    # 从用户输入获取名字
    name = input("请输入你的名字: ")
    print(greet(name))
