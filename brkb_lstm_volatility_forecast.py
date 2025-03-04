import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from pathlib import Path

# Step 1: Load dataset
f = Path("brkb.csv")
df = pd.read_csv(f, parse_dates=["date"])

# Step 2: Calculate realized volatility
df["ret"] = df["adjClose"].pct_change()
df["rv"] = df["ret"].rolling(20).std() * np.sqrt(252)  # Annualized realized volatility
df = df.dropna()
XX = df[["ret", "volume", "rv"]].values

# Step 3: Scale features
sc = MinMaxScaler()
XX_norm = sc.fit_transform(XX)

# Step 4: Generate LSTM sequences
seq_len = 20
X, Y = [], []
for i in range(seq_len, len(XX_norm)):
    X.append(XX_norm[i-seq_len:i])
    Y.append(XX_norm[i, 2])  # Target: realized volatility
X, Y = np.array(X), np.array(Y)

# Step 5: Split into training and testing sets
tr_split = int(0.8 * len(X))
X_tr, X_ts = X[:tr_split], X[tr_split:]
Y_tr, Y_ts = Y[:tr_split], Y[tr_split:]
X_tr = torch.FloatTensor(X_tr)
Y_tr = torch.FloatTensor(Y_tr)
X_ts = torch.FloatTensor(X_ts)
Y_ts = torch.FloatTensor(Y_ts)

# Step 6: Define LSTM architecture
class LSTM_RV(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers, out_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(in_dim, hid_dim, n_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hid_dim, out_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hid_dim).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hid_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Step 7: Initialize model and training components
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = LSTM_RV(in_dim=3, hid_dim=50, n_layers=2, out_dim=1).to(dev)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(m.parameters(), lr=0.001)
X_tr, Y_tr = X_tr.to(dev), Y_tr.to(dev)
X_ts = X_ts.to(dev)

# Step 8: Train LSTM model
n_epochs = 100
for ep in range(n_epochs):
    m.train()
    Y_pred_tr = m(X_tr)
    opt.zero_grad()
    loss = loss_fn(Y_pred_tr, Y_tr.unsqueeze(1))
    loss.backward()
    opt.step()
    if (ep + 1) % 10 == 0:
        print(f"Epoch {ep+1}/{n_epochs}, Loss: {loss.item():.6f}")

# Step 9: Generate predictions
m.eval()
with torch.no_grad():
    Y_pred_ts = m(X_ts).cpu().numpy()

# Step 10: Rescale predictions and actuals
Y_ts_raw = sc.inverse_transform(np.c_[np.zeros((len(Y_ts), 2)), Y_ts])[:, 2]
Y_pred_raw = sc.inverse_transform(np.c_[np.zeros((len(Y_pred_ts), 2)), Y_pred_ts])[:, 2]

# Step 11: Visualize realized volatility
dt = df["date"].iloc[tr_split + seq_len:]
fig = go.Figure()
fig.add_trace(go.Scatter(x=dt, y=Y_ts_raw, mode="lines", name="Actual RV", line=dict(color="#FF6B6B")))
fig.add_trace(go.Scatter(x=dt, y=Y_pred_raw, mode="lines", name="Forecast RV", line=dict(color="#4ECDC4", dash="dash")))
fig.update_layout(
    title="Realized Volatility Forecast for $BRKB (LSTM)",
    xaxis_title="Date",
    yaxis_title="Annualized Realized Volatility",
    template="plotly_dark",
    plot_bgcolor="rgb(40, 40, 40)",
    paper_bgcolor="rgb(40, 40, 40)",
    font=dict(color="white"),
    xaxis=dict(gridcolor="rgba(255, 255, 255, 0.1)", gridwidth=0.5),
    yaxis=dict(gridcolor="rgba(255, 255, 255, 0.1)", gridwidth=0.5),
    margin=dict(l=50, r=50, t=50, b=50),
    showlegend=True
)
fig.show()

# Step 12: Evaluate performance
rmse = np.sqrt(np.mean((Y_ts_raw - Y_pred_raw) ** 2))
print(f"Test RMSE: {rmse:.4f}")