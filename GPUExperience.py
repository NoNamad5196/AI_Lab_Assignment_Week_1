import torch
import torch.nn as nn

# CUDA(GPU) 사용 가능 여부 확인 후, device를 'cuda'로 설정
assert torch.cuda.is_available(), "GPU를 사용할 수 없습니다! CUDA 설정을 확인하세요."
device = torch.device("cuda")

print(f"메인 장치로 '{device}'를 사용합니다.")

# 생성 후 CPU로 보내기
tensor_cpu = torch.randn(3, 3)
tensor_gpu = tensor_cpu.to(device)
print(f"'to' 사용 후 텐서 위치: {tensor_gpu.device}")

# 처음부터 GPU에 생성하기
tensor_direct_gpu = torch.ones(3, 3, device=device)
print(f"'device=' 사용 후 텐서 위치: {tensor_direct_gpu.device}")

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

#모델 인스턴스 생성 후 GPU로 이동
model = MyModel().to(device)

print(f"모델 패러미터 위치: {next(model.parameters()).device}")

#GPU에 입력데이터 준비
input_tensor = torch.randn(10, 10, device=device)
#GPU에 있는 모델에 GPU에 있는 데이터를 입력
print(">>>>> 모델에 들어가기 직전 텐서 모양:", input_tensor.shape)
output = model(input_tensor)

#결과 확인
print(f"입력 텐서 위치: {input_tensor.device}")
print(f"출력 텐서 위치: {output.device}")