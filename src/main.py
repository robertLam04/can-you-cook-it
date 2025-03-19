import preprocess
from sklearn.model_selection import train_test_split
import train

x, y = preprocess.load_dataset("../recipe_ratings.csv", "../scraped_recipes.csv")

bow_features = preprocess.preprocess_bow(x)

X_train, X_test, y_train, y_test = train_test_split(bow_features, y, test_size=0.3, random_state=5)

trained_model = train.train_logistic_regression(X_train, y_train)
trained_model_metrics = train.evaluate_model(trained_model, X_test, y_test)

print("\nModel Performance (Linear Regression BoW):")
for metric, value in trained_model_metrics.items():
    print(f"{metric}: {value:.4f}")

average = train.AverageModel()
average.fit(X_train, y_train)
average_model_metrics = train.evaluate_model(average, X_test, y_test)

print("\nModel Performance (Average):")
for metric, value in average_model_metrics.items():
    print(f"{metric}: {value:.4f}")

random = train.RandomModel()
random_model_metrics = train.evaluate_model(random, X_test, y_test)

print("\nModel Performance (Random):")
for metric, value in random_model_metrics.items():
    print(f"{metric}: {value:.4f}")
