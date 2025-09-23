import torch
import torch.optim as optim

#1. 가상 데이터 생성 y = 2x + 1이라는 데이터로 구성
x_train = torch.FloatTensor([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y_train = torch.FloatTensor([[3.2],[5.1],[6.8],[9.3],[11],[12.8],[15.1],[17.2],[19],[21.3]])

#2. 모델의 가중치(W), 편향(b) 초기화
#requires_grad=True로 이 변수들이 학습을 통해서 계속 업데이트 되어야한다는것을 의미
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

#3. 옵티마이저 설정
optimizer = optim.SGD([W,b], lr=0.01)

#4. 훈련 루프 실행
nb_epochs = 3000
print("훈련을 시작합니다...")
for epoch in range(nb_epochs + 1):
    #1단계 예측
    #현재 가중치 W와 편향 b를 사용하여, 예측값을 계산, H(x) = Wx + b
    hypothesis = x_train * W + b

    #2단계 손실 계산
    #예측값과 실제값의 차이를 계산 , 평균 제곱 오차를 사용함
    cost = torch.mean((hypothesis - y_train) ** 2)

    #3단계 그래디언트 초기화
    #역전파를 수행하기 전에 이전 단계에 계산된 그래디언트 값을 0으로 초기화 합니다. 이래야 이전 계산값이 누적되지 않아 학습이 원할해집니다.
    optimizer.zero_grad()

    #4단계 역전파
    #손실에 대해서 각 가중치의 그래디언트를 계산
    #이 그래디언트는 가중치를 어느 방향으로 업데이트 해야 손실이 줄어드는지 알수 있음.
    cost.backward()

    #5단계 가중치 업데이트
    #계산된 그래디언트와 학습률을 사용하여 W와 b를 업데이트합니다.
    optimizer.step()

    #100번 마다 중간 과정 출력
    if epoch % 100 == 0:
        print(f'Epoch {epoch:4d}/{nb_epochs} | W: {W.item(): .3f} | b: {b.item(): .3f} | Cost: {cost.item(): .6f}')

print("\n훈련이 완료되었습니다.")
print(f"최종적으로 학습된 W: {W.item(): .3f} | b: {b.item(): .3f}")    
#학습된 모델로 새로운 값 예측해보기
new_x = 10
prediction = new_x * W + b
print(f"\nx = {new_x}일 때의 예측값: {prediction.item():.3f}")