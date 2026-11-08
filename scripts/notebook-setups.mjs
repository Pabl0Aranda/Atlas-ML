// The code on each page is a short excerpt: it uses X, y, df or a trained model
// without creating them. These cells make each notebook run on its own, with data
// shaped like the page's sandbox. `install` runs first, `setup` before the page's
// code, and `demo` after it, for excerpts that only define a function.

const moons = `from sklearn.datasets import make_moons

# Two interleaving half-moons, like the sandbox's Moons dataset.
X, y = make_moons(n_samples=300, noise=0.25, random_state=7)`;

const split = `from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)`;

const blobs = `from sklearn.datasets import make_blobs

# Four round groups of points, like the sandbox's Blobs dataset.
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=7)`;

const tightMoons = `from sklearn.datasets import make_moons

# Two clean half-moons: density and linkage methods can follow their shape.
X, _ = make_moons(n_samples=300, noise=0.05, random_state=7)`;

const trend = `import numpy as np

# A noisy straight-line trend with one input.
rng = np.random.default_rng(7)
X = rng.uniform(-1, 1, size=(100, 1))
y = 0.7 * X[:, 0] + 0.1 + rng.normal(0, 0.1, size=100)`;

const curve = `import numpy as np

# A noisy curve with one input, which a straight line can't follow.
rng = np.random.default_rng(7)
X = rng.uniform(-1, 1, size=(80, 1))
y = 0.8 * X[:, 0] ** 3 - 0.5 * X[:, 0] + rng.normal(0, 0.1, size=80)`;

const manyFeatures = `from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# 20 inputs, only 5 of which matter: room for a penalty to make a difference.
X, y = make_regression(n_samples=200, n_features=20, n_informative=5, noise=10, random_state=7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)`;

const validation = `from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# A model trained on part of the data, with predictions on held-out validation data.
X, y = make_moons(n_samples=300, noise=0.25, random_state=7)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
model = LogisticRegression().fit(X_train, y_train)
predictions = model.predict(X_val)
probabilities = model.predict_proba(X_val)`;

/** @type {Record<string, { install?: string, setup?: string, demo?: string }>} */
export const SETUPS = {
	'algorithms/adaboost': { setup: moons, demo: 'print(f"Training accuracy: {model.score(X, y):.1%}")' },
	'algorithms/dbscan': {
		setup: tightMoons,
		demo: 'print("Clusters:", len(set(labels) - {-1}), "| noise points:", (labels == -1).sum())',
	},
	'algorithms/decision-tree': { setup: moons },
	'algorithms/deep-q-networks': {
		demo: `# One training step on a random batch of 8 transitions, with 4 state values and 2 actions.
online, target = nn.Linear(4, 2), nn.Linear(4, 2)
target.load_state_dict(online.state_dict())
optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
batch = (torch.randn(8, 4), torch.randint(0, 2, (8,)), torch.randn(8), torch.randn(8, 4), torch.zeros(8))
print(train_step(online, target, optimizer, batch))`,
	},
	'algorithms/elastic-net': { setup: manyFeatures },
	'algorithms/gaussian-mixture-models': { setup: blobs },
	'algorithms/gaussian-process-regression': {
		setup: `import numpy as np

# A few noisy samples of a smooth curve, and a fine grid to predict on.
rng = np.random.default_rng(7)
X_train = rng.uniform(-1, 1, size=(12, 1))
y_train = np.sin(3 * X_train[:, 0]) + rng.normal(0, 0.05, size=12)
X_test = np.linspace(-1, 1, 50).reshape(-1, 1)`,
		demo: `print(mean[:5], std[:5])  # confident near data, unsure between points`,
	},
	'algorithms/gradient-boosting': { setup: `${moons}\n\n${split}` },
	'algorithms/hierarchical-clustering': { setup: tightMoons },
	'algorithms/k-means': { setup: blobs },
	'algorithms/knn': { setup: moons },
	'algorithms/lasso-regression': { setup: manyFeatures },
	'algorithms/linear-discriminant-analysis': {
		setup: `${moons}\n\n${split}`,
		demo: `print(X_projected.shape)  # two classes, so one dimension
print(f"Test accuracy: {model.score(X_test, y_test):.1%}")`,
	},
	'algorithms/linear-regression': { setup: trend },
	'algorithms/logistic-regression': { setup: `${moons}\n\n${split}` },
	'algorithms/mlp': {
		setup: `import torch
from sklearn.datasets import make_moons

# The Moons data as PyTorch tensors.
X_numpy, y_numpy = make_moons(n_samples=300, noise=0.2, random_state=7)
X = torch.tensor(X_numpy, dtype=torch.float32)
y = torch.tensor(y_numpy, dtype=torch.float32)`,
		demo: `with torch.no_grad():
    accuracy = ((model(X).squeeze(1) > 0) == y.bool()).float().mean()
print(f"Training accuracy: {accuracy:.1%}")`,
	},
	'algorithms/naive-bayes': { setup: `${moons}\n\n${split}` },
	'algorithms/pca': {
		setup: `import numpy as np

# A stretched, tilted cloud: most of the spread lies along one direction.
rng = np.random.default_rng(7)
X = rng.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], size=300)`,
	},
	'algorithms/perceptron': { setup: moons },
	'algorithms/polynomial-regression': { setup: curve },
	'algorithms/proximal-policy-optimization': {
		demo: `# The loss for a batch of 16 actions with random log-probabilities and advantages.
new_log_probs = torch.randn(16) * 0.1 - 1
old_log_probs = new_log_probs.detach() + torch.randn(16) * 0.1
print(ppo_loss(new_log_probs, old_log_probs, torch.randn(16), torch.rand(16)))`,
	},
	'algorithms/q-learning': {
		demo: `# Two states with two actions each. Taking action 1 from "start" reaches the goal for a reward of -1.
Q = {"start": [0.0, 0.0], "goal": [0.0, 0.0]}
q_learning_update(Q, "start", 1, r=-1, s_next="goal", done=True)
print(Q)  # the value of that action moves halfway to -1`,
	},
	'algorithms/random-forest': { setup: moons },
	'algorithms/ridge-regression': { setup: manyFeatures },
	'algorithms/sarsa': {
		demo: `# Two states with two actions each. Taking action 1 from "start" reaches the goal for a reward of -1.
Q = {"start": [0.0, 0.0], "goal": [0.0, 0.0]}
sarsa_update(Q, "start", 1, r=-1, s_next="goal", a_next=0, done=True)
print(Q)  # the value of that action moves halfway to -1`,
	},
	'algorithms/svm': { setup: moons },
	'concepts/overfitting-underfitting': { setup: moons },
	'data-preparation/cleaning': {
		setup: `import pandas as pd

# A small table with the usual problems: a duplicate row, inconsistent spelling,
# a missing value and an outlier price.
df = pd.DataFrame({
    "country": [" Spain", "spain", "France ", "france", "Spain", " Spain", "Germany", None],
    "price": [12.5, 13.0, 11.0, 12.0, 250.0, 12.5, None, 14.0],
})`,
		demo: `print(outliers)`,
	},
	'data-preparation/overview': {
		setup: `import numpy as np
import pandas as pd

# Customers with numbers, categories and some missing incomes.
rng = np.random.default_rng(7)
n = 300
X = pd.DataFrame({
    "age": rng.integers(18, 80, n).astype(float),
    "income": rng.normal(40_000, 12_000, n).round(),
    "plan": rng.choice(["basic", "plus", "pro"], n),
    "country": rng.choice(["es", "fr", "de"], n),
})
X.loc[rng.choice(n, 20, replace=False), "income"] = np.nan
y = ((X["plan"] != "basic") & (X["age"] > 30)).astype(int)
numeric_columns = ["age", "income"]
category_columns = ["plan", "country"]`,
	},
	'data-preparation/processing': {
		setup: `import numpy as np

# Two features on very different scales, and a column of color names.
rng = np.random.default_rng(7)
X_train = rng.normal([50, 0.5], [20, 0.1], size=(200, 2))
X_test = rng.normal([50, 0.5], [20, 0.1], size=(50, 2))
colors_train = [["red"], ["green"], ["blue"], ["green"]]`,
		demo: `print(X_train_scaled.mean(axis=0).round(2), X_train_scaled.std(axis=0).round(2))
print(encoder.transform([["blue"], ["purple"]]).toarray())  # an unseen color becomes all zeros`,
	},
	'data-preparation/imbalanced-data': {
		install: `%pip install --quiet imbalanced-learn`,
		setup: `import numpy as np

# Two overlapping clouds of 1,000 samples, one in ten from the rare class, like the sandbox.
rng = np.random.default_rng(7)
y = (np.arange(1000) % 10 == 0).astype(int)
centers = np.where(y[:, None] == 1, 0.45, -0.15)
X = centers + rng.normal(0, 0.24, size=(1000, 2))
print("Rare samples:", y.sum(), "of", len(y))`,
		demo: `# The same model without rebalancing, for comparison. Look at recall for class 1.
plain_model = LogisticRegression().fit(X_train, y_train)
print(classification_report(y_test, plain_model.predict(X_test), digits=3))`,
	},
	'data-preparation/splitting': {
		setup: `${moons}

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)`,
	},
	'distance-metrics/direction-and-shape': {
		setup: `import numpy as np

# Correlated data, where the Mahalanobis distance differs from the Euclidean one.
rng = np.random.default_rng(7)
X = rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], size=300)`,
	},
	'distance-metrics/distributions': {},
	'distance-metrics/geometric': {},
	'distance-metrics/sequences': {
		install: `%pip install --quiet rapidfuzz dtaidistance`,
		setup: `import numpy as np

# The same wave, once shifted by 10 steps.
t = np.linspace(0, 4 * np.pi, 100)
series_a = np.sin(t)
series_b = np.sin(t - 0.5)`,
	},
	'distance-metrics/sets-and-categories': {},
	'testing/classification-metrics': {
		setup: `${moons}

${split}

from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(X_train, y_train)`,
	},
	'testing/explainability': {
		install: `%pip install --quiet shap lime`,
		setup: `from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 569 tumors described by 30 measurements, each labeled malignant or benign.
data = load_breast_cancer()
X, y = data.data, data.target
feature_names, class_names = list(data.feature_names), list(data.target_names)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
model = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_train, y_train)
print(f"Test accuracy: {model.score(X_test, y_test):.1%}")`,
	},
	'testing/regression-metrics': {
		setup: `from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=200, n_features=3, noise=15, random_state=7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
model = LinearRegression().fit(X_train, y_train)`,
	},
	'validation/cross-validation': { setup: `${moons}\n\n${split}` },
	'validation/hyperparameter-tuning': { setup: `${moons}\n\n${split}` },
	'validation/indicators': { setup: validation },
	'validation/validation-matrices': { setup: validation },
};
