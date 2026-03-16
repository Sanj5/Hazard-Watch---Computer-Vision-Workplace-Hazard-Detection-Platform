import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def build_synthetic_dataset(n=3000):
    rng = np.random.default_rng(42)
    ppe = rng.uniform(0, 1, n)
    behavior = rng.uniform(0, 1, n)
    near_miss = rng.uniform(0, 1, n)
    collision = rng.uniform(0, 1, n)

    risk_signal = 0.35 * ppe + 0.25 * behavior + 0.2 * near_miss + 0.2 * collision
    noise = rng.normal(0, 0.07, n)
    y = ((risk_signal + noise) > 0.55).astype(int)

    x = np.column_stack([ppe, behavior, near_miss, collision])
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3000)
    args = parser.parse_args()

    x, y = build_synthetic_dataset(args.samples)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

    model = LogisticRegression(max_iter=500)
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    print(classification_report(y_test, preds))
    print("Coefficients:", model.coef_)


if __name__ == "__main__":
    main()
