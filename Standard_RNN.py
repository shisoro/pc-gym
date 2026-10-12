import torch
import torch.nn as nn

class StandardRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.W_hy = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        h = torch.tanh(self.W_xh(x) + self.W_hh(h))
        y = self.W_hy(h)
        return y, h
    
model = StandardRNN(
    input_size = 1,
    hidden_size = 3,
    output_size = 1
)

# 学習率0.01の確率的勾配降下法
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 入力値, 目標値の初期設定
Lx = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5]])
target = torch.tensor([[0.6]])

for epoch in range(100):
    # 勾配を初期化
    optimizer.zero_grad()

    # 各学習回の前に隠れ状態を初期化
    h = torch.zeros(1, 3)

    # 時系列データを入力
    for t,x in enumerate(Lx):
        x = x.unsqueeze(0)
        # 順伝播
        y, h = model(x, h)

    # 損失計算（MSE）
    loss = (y - target).pow(2).mean()

    # back propagation
    loss.backward()

    # 重みを更新
    optimizer.step()

    if epoch % 10 == 0:
        print(f"epoch={epoch}, loss={loss.item():.6f}")
        print("prediction=", y)