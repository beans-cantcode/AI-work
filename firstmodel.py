import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/benqin/Downloads/House_price.csv")
df.dropna(inplace=True)
df.drop(columns=['Address'], inplace=True)
print(df.columns.tolist())
price_series = df['Price']
df.drop(columns=['Price'], inplace=True)
print(df.info())
print(df.shape)
print(df.head())

house_data = torch.tensor(df.values, dtype=torch.float32)
price = torch.tensor(price_series.values, dtype=torch.float32).unsqueeze(1)

print(house_data[:5])
print(price[:5])

X_train, X_test, y_train, y_test = train_test_split(house_data, price, test_size=0.2, random_state=42)

X = house_data
y = price

X_mean = X.mean(dim=0)
X_std = X.std(dim=0)

y_mean = y.mean(dim=0)
y_std = y.std(dim=0)

Xt_scaled = (X_train - X_mean) / X_std
yt_scaled = (y_train - y_mean) / y_std
XX_scaled = (X_test - X_mean) / X_std
yy_scaled = (y_test - y_mean) / y_std

#print(x[:5])

#true_W = torch.tensor ([[2.0]])
#true_b = torch.tensor ([1.0])

#true_y = x @ true_W + true_b + torch.randn(N, D_out) * 0.1


class LinRegMod(nn.Module):
    def __init__(self, D_in, D_out):
        super().__init__()

        self.linear_layer = nn.Linear(D_in, D_out)
    
    def forward(self, x):
        return self.linear_layer(x)
    

model = LinRegMod(5, 1)
optimzer = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

for epoch in range(1000):
    
    yguess = model.forward(Xt_scaled)
    loss = loss_fn(yguess, yt_scaled)
    
    optimzer.zero_grad()
 
    loss.backward()
    
    optimzer.step()
    if epoch % 100 == 0:
        print("Epoch:", epoch)
        print("Y guess" , yguess[:3])
        print("loss:", loss)
        print("weights:", model.linear_layer.weight, model.linear_layer.bias)
    
print("Final weights:", model.linear_layer.weight, model.linear_layer.bias)

with torch.no_grad():
    pred_scaled = model(XX_scaled)

    # Convert predictions back into real dollar prices
    pred_prices = pred_scaled * y_std + y_mean

    print("Predicted prices:")
    print(pred_prices[:5])

    print("Real prices:")
    print(y_test[:5])

    rmse = torch.sqrt(loss_fn(pred_prices, y_test))
    print("RMSE:", rmse.item())

"""
#first model

w = torch.randn(D_in, D_out, requires_grad=True)
b = torch.randn(1, requires_grad=True)

print("Initial weights:", w, b)

for t in range(50):
    y_pred = x @ w + b

    print("Predicted y:", y_pred[:3]," True y:", true_y[:3])

    loss = (y_pred - true_y).pow(2).sum()/N

    print("loss:",loss)

    loss.backward()
    with torch.no_grad():
        w -= 0.1 * w.grad
        b -= 0.1 * b.grad
        w.grad.zero_()
        b.grad.zero_()

print("Final weights:", w, b)
print("True weights:", true_W, true_b)
"""


"""
linear_layer = torch.nn.Linear(D_in, D_out)

#print(linear_layer.weight)
#print(linear_layer.bias)

y_prednn = linear_layer(x)

#print(y_prednn[:3])

relu = torch.nn.ReLU()
gelu = torch.nn.GELU()
softmax = torch.nn.Softmax(dim=1)

vocab_size = 10
embedding_dim= 3

embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)

input_ids = torch.tensor([1, 2, 3, 4])
word_embeddings = embedding_layer(input_ids)

#print(input_ids, word_embeddings)

norm_layer = torch.nn.LayerNorm(embedding_dim)

dropout_layer = torch.nn.Dropout(p=0.5)
"""