import torch
import torch.nn as nn


# ==========================================================
# Raw EEG Encoder (Paper Branch)
# ==========================================================

class KernelEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv1d(
                in_channels=128,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(64),

            nn.GELU(),

            nn.Conv1d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(128),

            nn.GELU(),

            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):

        x = self.network(x)

        x = x.squeeze(-1)

        return x


# ==========================================================
# Differential Entropy Encoder
# ==========================================================

class DEEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(640, 256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 128)
        )

    def forward(self, x):

        return self.network(x)


# ==========================================================
# EMG Encoder
# ==========================================================

class EMGEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(12, 32),

            nn.ReLU(),

            nn.Linear(32, 16)
        )

    def forward(self, x):

        return self.network(x)


# ==========================================================
# EOG Encoder
# ==========================================================

class EOGEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(6, 16),

            nn.ReLU(),

            nn.Linear(16, 8)
        )

    def forward(self, x):

        return self.network(x)


# ==========================================================
# Attention Layer
# ==========================================================

class Attention(nn.Module):

    def __init__(self, hidden_size):

        super().__init__()

        self.score = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):

        weights = torch.softmax(
            self.score(x),
            dim=1
        )

        context = torch.sum(
            weights * x,
            dim=1
        )

        return context


# ==========================================================
# Complete Network
# ==========================================================

class SilentSpeechDecoder(nn.Module):

    def __init__(
        self,
        num_classes,
        hidden_size=256,
        num_layers=2
    ):

        super().__init__()

        self.kernel = KernelEncoder()

        self.de = DEEncoder()

        self.emg = EMGEncoder()

        self.eog = EOGEncoder()

        fused_size = 128 + 128 + 16 + 8

        self.projection = nn.Sequential(

            nn.Linear(fused_size, 256),

            nn.ReLU(),

            nn.Dropout(0.3)
        )

        self.lstm = nn.LSTM(

            input_size=256,

            hidden_size=hidden_size,

            num_layers=num_layers,

            bidirectional=True,

            batch_first=True,

            dropout=0.3
        )

        self.attention = Attention(hidden_size)

        self.classifier = nn.Sequential(

            nn.Linear(hidden_size * 2, 256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(
        self,
        eeg,
        entropy,
        emg,
        eog
    ):

        eeg = self.kernel(eeg)

        entropy = self.de(entropy)

        emg = self.emg(emg)

        eog = self.eog(eog)

        fused = torch.cat(

            [
                eeg,
                entropy,
                emg,
                eog
            ],

            dim=1
        )

        fused = self.projection(fused)

        fused = fused.unsqueeze(1)

        lstm_out, _ = self.lstm(fused)

        context = self.attention(lstm_out)

        output = self.classifier(context)

        return output


# ==========================================================
# Standalone Test
# ==========================================================

if __name__ == "__main__":

    batch = 8

    model = SilentSpeechDecoder(
        num_classes=100
    )

    eeg = torch.randn(
        batch,
        128,
        900
    )

    entropy = torch.randn(
        batch,
        640
    )

    emg = torch.randn(
        batch,
        12
    )

    eog = torch.randn(
        batch,
        6
    )

    output = model(
        eeg,
        entropy,
        emg,
        eog
    )

    print("=" * 50)

    print(model)

    print("=" * 50)

    print("EEG Input      :", eeg.shape)
    print("Entropy Input  :", entropy.shape)
    print("EMG Input      :", emg.shape)
    print("EOG Input      :", eog.shape)

    print("Prediction     :", output.shape)

    print("=" * 50)