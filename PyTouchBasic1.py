import torch

#리스트로 텐서를 생성함.
data = [[1,2],[3,4]]
x_data = torch.tensor(data)

#넘파이 배열로부터 텐서를 생성함.
import numpy as np
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones(5, 3) # 5x3 크기의 1로 채워진 텐서
print(f"Ones Tensor: \n {x_ones} \n")

x_zeros = torch.zeros(5, 3) # 5x3 크기의 0으로 채워진 텐서
print(f"Zeros Tensor: \n {x_zeros} \n")

x_rand = torch.rand(5, 3) # 5x3 크기의 랜덤 텐서
print(f"Random Tensor: \n {x_rand} \n")

#텐서의 차원 확인
print(f"Shape of x_ones: {x_ones.shape}")
print(f"Shape of x_zeros: {x_zeros.shape}")
print(f"Shape of x_rand: {x_rand.shape}")