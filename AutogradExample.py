import torch
#1. 텐서 생성
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
print(f"x: {x.item()}일때, y: {y.item()}의 미분값은 {x.grad.item()}입니다.")