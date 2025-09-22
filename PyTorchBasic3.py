import torch
# 브로드캐스팅 스칼라 값의 연산
x = torch.tensor([1,2,3])
y = 5
z = x + y # torch_add(x,y) 함수로 변형가능
print(f"브로드캐스팅_1(스칼라) : {z}")
# 브로드캐스팅 벡터 값의 연산
a = torch.tensor([1,2,3])
b = torch.tensor([[10],[20]])
c = a + b # a는([1,2,3],[1,2,3]) b는 ([10,10,10],[20,20,20]) 브로드캐스팅 됨.
print(f"브로드캐스팅_2(벡터) : {c}")

#집계 연산
tensor_2d = torch.tensor([[1,2,3],[4,5,6]])
sum_all = torch.sum(tensor_2d) # 2차원 텐서의 모든 원소의 합
print(f"전체 합계: {sum_all}")
sum_dim0 = torch.sum(tensor_2d, dim=0) #행의 합계 (dim=0) [1+4, 2+5, 3+6]
print(f"행의 합계: {sum_dim0}")
sum_dim1 = torch.sum(tensor_2d, dim=1) #열의 합계 (dim=1) [1+2+3, 4+5+6]
print(f"열의 합계: {sum_dim1}")
mean_all = torch.mean(tensor_2d.float()) # 2차원 텐서의 모든 원소의 평균
print(f"전체 평균: {mean_all}")

#텐서 조작
tensor_a = torch.arange(1, 7).reshape(2, 3) # 1부터 6까지의 숫자를 2x3 크기의 텐서로 변환
print(f"원래 텐서: {tensor_a}")
#모양 변경하기
#-1은 자동으로 차원을 계산함.
reshaped_a = tensor_a.reshape(3, -1) # [[1,2,3],[4,5,6]] -> [[1,2],[3,4],[5,6]]
print(f"모양 변경 후 텐서: {reshaped_a}")
#텐서 합치기
newX = torch.tensor([[1,2],[3,4]])
newY = torch.tensor([[5,6],[7,8]])
#dim = 0 (세로로 합치기)
stacked_vertical = torch.cat([newX,newY],dim=0)
print(f"세로로 합친 텐서: {stacked_vertical}")
#dim = 1 (가로로 합치기)
stacked_horizontal = torch.cat([newX,newY],dim=1)
print(f"가로로 합친 텐서: {stacked_horizontal}")
#텐서 분할하기
split_tensor = torch.tensor([1,2,3,4,5,6])
split_tensor_1, split_tensor_2 = torch.split(split_tensor, [2,4], dim=0)
print(f"분할 후 텐서_1: {split_tensor_1}")
print(f"분할 후 텐서_2: {split_tensor_2}")