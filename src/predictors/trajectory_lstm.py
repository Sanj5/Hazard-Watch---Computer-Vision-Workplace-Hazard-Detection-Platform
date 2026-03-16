from typing import List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:
    class TrajectoryLSTM(nn.Module):
        def __init__(self, input_size: int = 2, hidden_size: int = 64, num_layers: int = 2, pred_steps: int = 8):
            super().__init__()
            self.pred_steps = pred_steps
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, pred_steps * 2),
            )

        def forward(self, x):
            output, _ = self.lstm(x)
            last = output[:, -1, :]
            y = self.head(last)
            return y.view(-1, self.pred_steps, 2)


class TrajectoryForecaster:
    """Forecast future centers; uses LSTM model when available, otherwise linear extrapolation."""

    def __init__(self, pred_steps: int = 8, model_path: Optional[str] = None):
        self.pred_steps = pred_steps
        self.model = None

        if torch is not None and nn is not None:
            self.model = TrajectoryLSTM(pred_steps=pred_steps)
            if model_path:
                state = torch.load(model_path, map_location="cpu")
                self.model.load_state_dict(state)
            self.model.eval()

    @staticmethod
    def _fallback_linear(history: List[Tuple[float, float]], pred_steps: int) -> List[Tuple[float, float]]:
        if len(history) < 2:
            return [history[-1] if history else (0.0, 0.0)] * pred_steps

        (x1, y1), (x2, y2) = history[-2], history[-1]
        vx, vy = x2 - x1, y2 - y1
        preds = []
        cx, cy = x2, y2
        for _ in range(pred_steps):
            cx += vx
            cy += vy
            preds.append((cx, cy))
        return preds

    def predict(self, history: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(history) < 2:
            return self._fallback_linear(history, self.pred_steps)

        if self.model is None or torch is None:
            return self._fallback_linear(history, self.pred_steps)

        seq = history[-12:]
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y = self.model(x)[0].cpu().numpy().tolist()
        return [(float(px), float(py)) for px, py in y]
