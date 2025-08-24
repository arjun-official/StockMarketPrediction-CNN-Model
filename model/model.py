
from torch import nn
import torch


class CNNStocksModule(nn.Module):
    """
    Hybrid 1D-CNN + LSTM for time-series regression.
    Input x: tensor of shape [batch, seq_len]
    Output: tensor of shape [batch] (next-steps return/regression)
    """
    def __init__(self, window_length: int,
                 conv_channels: int = 32,
                 conv_kernel_size: int = 5,
                 lstm_hidden: int = 64,
                 lstm_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        assert window_length >= conv_kernel_size, "window_length must be >= conv_kernel_size"

        # 1D CNN extracts local patterns
        self.conv = nn.Conv1d(in_channels=1,
                              out_channels=conv_channels,
                              kernel_size=conv_kernel_size,
                              padding=0,
                              bias=True)
        self.bn = nn.BatchNorm1d(conv_channels)
        self.act = nn.ReLU()

        # LSTM captures temporal dependencies across the convolved features
        # After conv with kernel k (no padding), seq_len becomes window_length - k + 1
        self.seq_after_conv = window_length - conv_kernel_size + 1
        self.lstm = nn.LSTM(input_size=conv_channels,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            dropout=dropout if lstm_layers > 1 else 0.0,
                            batch_first=True,
                            bidirectional=False)

        # Head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len]
        """
        # Add channel dimension for Conv1d: [batch, 1, seq_len]
        x = x.unsqueeze(1)
        x = self.conv(x)                # [batch, C, seq']
        x = self.bn(x)
        x = self.act(x)
        # Prepare for LSTM: [batch, seq', C]
        x = x.transpose(1, 2)
        # LSTM
        x, (h_n, c_n) = self.lstm(x)    # x: [batch, seq', hidden]
        # Use last time step
        last = x[:, -1, :]              # [batch, hidden]
        out = self.dropout(last)
        out = self.fc(out).squeeze(-1)  # [batch]
        return out
