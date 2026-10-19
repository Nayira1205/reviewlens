# Glossary: terms used in this project, in plain English

**Accuracy**: share of reviews the model got right. Misleading when one class dominates (here about 78% of reviews are positive).

**Baseline**: a simple first model used as a reference. Better models must beat it.

**Bigram / n-gram**: a pair (or group) of neighbouring words, like "not good". Using them lets a simple model notice negation.

**Class imbalance**: when some labels are much rarer than others. Here: about 78% positive, 15% negative, 7 to 8% neutral.

**Classifier**: a model that puts an input into one of several classes (here: positive, neutral, negative).

**Confusion matrix**: a table where rows are the true labels and columns are the predictions. The diagonal shows correct answers; everything else is a mistake.

**Data leakage**: when information from the test set sneaks into training (for example duplicate rows in both), making scores look better than they really are.

**DataFrame**: a table in pandas (rows and columns), like an Excel sheet in Python.

**EDA (exploratory data analysis)**: looking at the data (sizes, missing values, distributions) before modelling.

**F1 score**: one number combining precision and recall. **Macro-F1** is the plain average of F1 over all classes, so a rare class counts as much as a common one.

**Label**: the "correct answer" for a review. Ours comes from the star rating.

**Label noise**: labels that are inconsistent. For example the text "good" given 3 stars by one person and 5 stars by another. No model can get both right.

**Lemmatization**: reducing a word to its dictionary form (batteries to battery, working to work).

**Logistic Regression**: a simple, fast classifier that learns a weight for each word. Easy to explain, so a good baseline.

**Negation**: words like "not" or "no" that flip meaning. We keep them during cleaning.

**Precision**: of the reviews the model called a class, the share that truly were that class.

**Recall**: of the reviews that truly are a class, the share the model found.

**Regular expression (regex)**: a pattern used to find or replace pieces of text, for example "any letter repeated 3 or more times".

**Stop words**: very common words (the, is, and) often removed because they carry little meaning. We do not remove negations.

**Stratified split**: splitting data so each part keeps the same class mix.

**Strict test set**: our name for test reviews whose exact text never appeared in training.

**TF-IDF**: a way to turn text into numbers. A word scores high if it appears often in a review but is rare across all reviews.

**Tokenization**: splitting text into words (tokens).

**Train / test split**: training data is what the model learns from; test data is held back to score it fairly.

**Vectorizer**: the tool that converts text into numbers (here `TfidfVectorizer`).

**VADER**: a rule-based tool that scores sentiment from a dictionary of words. Used by the dataset's author to create the `Sentiment` column, which we do not train on.
