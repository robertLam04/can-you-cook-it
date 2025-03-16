import preprocess
from sklearn.model_selection import train_test_split
import train

x, y = preprocess.load_dataset("../recipe_ratings.csv", "../scraped_recipes.csv")

bow_features = preprocess.preprocess_bow(x)

X_train, X_test, y_train, y_test = train_test_split(bow_features, y, test_size=0.2, random_state=42)

trained_model = train.train_logistic_regression(X_train, y_train)
trained_metrics = train.evaluate_model(trained_model, X_test, y_test)

print("Model Performance (Linear Regression BoW):")
for metric, value in trained_metrics.items():
    print(f"{metric}: {value:.4f}")

simple_model = train.AverageModel()
simple_model.fit(X_train, y_train)
simple_metrics = train.evaluate_model(simple_model, X_test, y_test)

print("Model Performance (Average):")
for metric, value in simple_metrics.items():
    print(f"{metric}: {value:.4f}")