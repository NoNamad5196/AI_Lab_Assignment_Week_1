import torch
# 텐서 생성
x = torch.tensor([10.,20.,30.])
y = torch.tensor([1.,2.,3.])

# 텐서 연산
# 덧셈
z_sum = x + y # torch_add(x,y) 함수로 변형가능
print(f"덧셈: {z_sum}")
# 뺄셈
z_sub = x - y # torch_sub(x,y)
print(f"뺄셈: {z_sub}")
# 곱셈
z_mul = x * y # torch_mul(x,y)
print(f"곱셈: {z_mul}")
# 나눗셈
z_div = x / y # torch_div(x,y)
print(f"나눗셈: {z_div}")
# 제곱
z_pow = x ** y # torch_pow(x,y)
print(f"제곱: {z_pow}")
# 절대값
x_abs = torch.abs(torch.tensor([-10.,20.,-30.]))
print(f"절댓값: {x_abs}")