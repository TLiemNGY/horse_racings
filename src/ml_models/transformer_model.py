import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader


def preprocess_transformer_data(df, label_encoders=None, scaler=None, is_train=True):
    """Convert categorical columns to indices and normalize numerical columns for Transformer."""

    categorical_cols = [
        "venue", "config", "surface", "going", "horse_ratings", "race_class", "horse_country",
        "horse_type", "horse_gear", "trainer_id", "jockey_id", "specialized_trainer"
    ]
    numerical_cols = [
        "horse_age", "declared_weight", "actual_weight", "draw", "win_odds", "place_odds",
        "total_horses", "global_weight"
    ]
    target_col = "relative_ranking"

    if is_train:
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    else:
        for col in categorical_cols:
            df[col] = df[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_
            else len(label_encoders[col].classes_))

    if is_train:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    else:
        df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df, categorical_cols, numerical_cols, target_col, label_encoders, scaler


class TransformerHorseRaceDataset(Dataset):
    def __init__(self, df, cat_cols, num_cols, target_col):
        self.cat_data = df[cat_cols].values
        self.num_data = df[num_cols].values.astype(float)
        self.targets = df[target_col].values.astype(float)
        self.race_id = df['race_id'].values

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.cat_data[idx], dtype=torch.long),
            torch.tensor(self.num_data[idx], dtype=torch.float),
            torch.tensor(self.targets[idx], dtype=torch.float),
            torch.tensor(self.race_id[idx], dtype=torch.long)
        )


class TransformerHorseRaceModel(nn.Module):
    def __init__(self, cat_sizes, embedding_dim, num_features, transformer_dim, num_heads, num_layers):
        super(TransformerHorseRaceModel, self).__init__()

        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, embedding_dim) for size in cat_sizes]
        )

        self.num_feature_projection = nn.Linear(num_features, embedding_dim)

        self.dropout = nn.Dropout(0.5)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim * len(cat_sizes) + embedding_dim,
            nhead=num_heads,
            dropout=0.3
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embedding_dim * len(cat_sizes) + embedding_dim, 1)

    def forward(self, cat_data, num_data):
        cat_embeds = [embed(cat_data[:, i]) for i, embed in enumerate(self.embeddings)]
        cat_embeds = torch.cat(cat_embeds, dim=-1)

        num_features = self.num_feature_projection(num_data)
        x = torch.cat([cat_embeds, num_features], dim=-1)

        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        x = self.fc(x)

        return x.squeeze(-1)


def train_transformer(df_train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔄 Modèle entraîné sur : {device}")

    df_train, cat_cols, num_cols, target_col, label_encoders, scaler = preprocess_transformer_data(df_train,
                                                                                                   is_train=True)

    cat_sizes = [len(le.classes_) + 1 for le in label_encoders.values()]
    dataset = TransformerHorseRaceDataset(df_train, cat_cols, num_cols, target_col)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TransformerHorseRaceModel(
        cat_sizes=cat_sizes,
        embedding_dim=32,
        num_features=len(num_cols),
        transformer_dim=256,
        num_heads=8,
        num_layers=3
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    def winner_loss(y_pred, y_true, race_id):
        loss = 0
        unique_races = race_id.unique()

        for r in unique_races:
            mask = (race_id == r)
            y_pred_race = y_pred[mask]
            y_true_race = y_true[mask]

            if len(y_true_race) == 0:
                continue

            winner_mask = (y_true_race == 1).float()
            winner_pred = (y_pred_race * winner_mask).sum()

            best_pred = y_pred_race.max()

            loss += ((best_pred - winner_pred) ** 2) * (1 + (1 - winner_pred) ** 2)

        return loss / len(unique_races)

    model.train()
    for epoch in range(10):
        epoch_loss = 0
        for cat_data, num_data, targets, race_id in dataloader:
            cat_data, num_data, targets, race_id = (
            cat_data.to(device), num_data.to(device), targets.to(device), race_id.to(device))
            optimizer.zero_grad()
            preds = model(cat_data, num_data)

            for r in race_id.unique():
                mask = (race_id == r)
                preds[mask] = torch.softmax(preds[mask], dim=0)

            loss = winner_loss(preds, targets, race_id)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader):.4f}")

    return model, label_encoders, scaler


def predict_transformer(df_test, model, label_encoders, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔄 Modèle entraîné sur : {device}")

    df_test, cat_cols, num_cols, target_col, _, _ = preprocess_transformer_data(df_test, label_encoders, scaler,
                                                                                is_train=False)
    dataset = TransformerHorseRaceDataset(df_test, cat_cols, num_cols, target_col)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model.eval()
    predictions = []
    with torch.no_grad():
        for cat_data, num_data, _, race_id in dataloader:
            cat_data, num_data = cat_data.to(device), num_data.to(device)
            preds = model(cat_data, num_data)

            for r in race_id.unique():
                mask = (race_id == r)
                preds[mask] = torch.softmax(preds[mask], dim=0)

            predictions.extend(preds.tolist())

    return predictions
