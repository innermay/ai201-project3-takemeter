# TakeMeter

TakeMeter is a text classification project for AI201 Project 3. The goal of this project is to classify public T-Mobile-related Reddit posts and comments based on the main purpose of the text.

The classifier uses four labels: `help_request`, `customer_complaint`, `employee_complaint`, and `deal_or_policy_info`. I collected public Reddit data using an Reddit scraper, prepared the dataset as a CSV, fine-tuned `distilbert-base-uncased` in Google Colab, and compared the fine-tuned model against a zero-shot Groq baseline using `llama-3.3-70b-versatile`.

---

## Community Choice

I chose public T-Mobile Reddit discussions as my online community. This community is a good fit for a classification task because the posts and comments are text-heavy and include many different kinds of discourse. Some users ask for help with billing, service, devices, or account issues. Others complain about customer service, coverage, or store experiences. Some posts come from an employee perspective, and others explain promotions, plans, or policy changes.

These distinctions matter because a reader would respond differently to each kind of post. A help request needs troubleshooting, a customer complaint shows frustration, an employee complaint reveals workplace experience, and a deal or policy post is mainly informational.

---

## Label Taxonomy

### `help_request`

A post is labeled `help_request` if the writer is asking for advice, troubleshooting, or clarification about a T-Mobile issue, such as billing, service, plans, devices, upgrades, promotions, or account problems.

Example 1:

> My bill went up after changing plans. Can someone explain what might have happened?

Example 2:

> I switched to eSIM and now my service keeps dropping. Has anyone fixed this before?

Decision rule: If the writer is asking how to fix something, I label it `help_request`, even if the tone sounds frustrated.

---

### `customer_complaint`

A post is labeled `customer_complaint` if the writer is mainly complaining from a customer perspective about T-Mobile service, billing, coverage, support, store experience, promotions, or account problems.

Example 1:

> T-Mobile support keeps transferring me and nobody can fix my account issue.

Example 2:

> My coverage has gotten worse in my area and I am tired of paying this much for unreliable service.

Decision rule: If the main purpose is venting or criticizing a customer experience, I label it `customer_complaint`.

---

### `employee_complaint`

A post is labeled `employee_complaint` if the writer is mainly complaining from an employee or worker perspective about store management, quotas, commission, scheduling, leadership, customers, job stress, or workplace expectations.

Example 1:

> The sales goals keep going up, but staffing keeps getting worse. It feels impossible to hit numbers and still help customers properly.

Example 2:

> Management keeps pushing add-ons even when customers clearly do not need them, and employees are the ones who get blamed when people complain.

Decision rule: If the writer is speaking from a worker perspective, such as mentioning their store, manager, quota, commission, or customers they serve, I label it `employee_complaint`.

---

### `deal_or_policy_info`

A post is labeled `deal_or_policy_info` if it mainly shares or explains information about T-Mobile promotions, plans, upgrades, trade-ins, pricing, policy changes, account rules, or eligibility.

Example 1:

> The new trade-in promo gives higher credits on Go5G Plus than on older plans.

Example 2:

> If you cancel a line too soon after adding a promotion, you may lose the monthly bill credits.

Decision rule: If the post mainly explains a rule, promotion, plan change, bill credit, or eligibility detail, I label it `deal_or_policy_info`.

---

## Data Collection and Labeling Process

I collected public T-Mobile-related Reddit posts and comments using an Reddit scraper. The raw export contained 400 rows. I used a preparation script, `prepare_dataset.py`, to combine useful fields such as title and body into one text field, remove deleted or very short rows, remove duplicates, and create the final dataset CSV.

The final dataset file is:

```text
data/takemeter_dataset.csv
```

The final CSV has exactly these columns:

```text
text,label,notes
```

The preparation script used rule-based and AI-assisted pre-labeling to speed up the annotation process. I treated those labels as a first pass and reviewed the dataset for obvious errors and borderline cases before training. I also kept notes for examples where the label boundary was unclear.

I did not include usernames, profile pages, account numbers, phone numbers, or private/internal T-Mobile information in the final dataset. The final dataset only keeps the text needed for classification, the label, and optional notes.

---

## Dataset Summary

The raw scraper export contained:

```text
400 raw rows
```

After cleaning and filtering, the final labeled dataset contained:

```text
233 labeled examples
```

Label distribution:

| Label                 | Count | Percentage |
| --------------------- | ----: | ---------: |
| `help_request`        |   113 |      48.5% |
| `deal_or_policy_info` |    98 |      42.1% |
| `customer_complaint`  |    12 |       5.2% |
| `employee_complaint`  |    10 |       4.3% |

No single label is above 70% of the dataset. However, the dataset is still imbalanced because `customer_complaint` and `employee_complaint` have far fewer examples than the other two labels. This imbalance likely affected model performance.

Skipped rows summary:

| Skip Reason           | Count |
| --------------------- | ----: |
| No clear label signal |   116 |
| Removed or too short  |    50 |
| Duplicate             |     1 |
| Total skipped         |   167 |

---

## Difficult-to-Label Examples

### Difficult Example 1

Text excerpt:

> My bill is wrong again and support has been useless. Does anyone know how to get this fixed?

Possible labels:

* `help_request`
* `customer_complaint`

Final decision:

```text
help_request
```

Why: The post sounds frustrated, but the main purpose is asking how to fix the issue.

---

### Difficult Example 2

Text excerpt:

> T-Mobile changed the autopay discount rules and now credit cards do not qualify anymore. This is frustrating.

Possible labels:

* `deal_or_policy_info`
* `customer_complaint`

Final decision:

```text
deal_or_policy_info
```

Why: The main content explains a policy change. The frustration is secondary.

---

### Difficult Example 3

Text excerpt:

> At my store, managers keep telling us to push certain add-ons because the commission structure rewards it more than basic upgrades.

Possible labels:

* `employee_complaint`
* `deal_or_policy_info`

Final decision:

```text
employee_complaint
```

Why: The post mentions store management and employee pressure, so the worker perspective is the strongest signal.

---

## Fine-Tuning Approach

I fine-tuned `distilbert-base-uncased` using the provided TakeMeter starter notebook in Google Colab with a T4 GPU runtime.

Training setup:

| Setting                         | Value                     |
| ------------------------------- | ------------------------- |
| Base model                      | `distilbert-base-uncased` |
| Platform                        | Google Colab              |
| Runtime                         | T4 GPU                    |
| Train / validation / test split | 70% / 15% / 15%           |
| Epochs                          | 3                         |
| Learning rate                   | 2e-5                      |
| Batch size                      | 16                        |
| Max token length                | 256                       |

I kept the default training settings because the dataset was small. Three epochs gave the model a chance to learn the label patterns without intentionally over-training on a small dataset. If I were improving this project, I would collect more balanced data before changing hyperparameters.

---

## Baseline Approach

For the baseline, I used Groq with `llama-3.3-70b-versatile` as a zero-shot classifier. The baseline prompt included the four label definitions, one example per label, and decision rules for ambiguous cases. The model was instructed to output only the exact label name.

The baseline was run on the same test set as the fine-tuned DistilBERT model.

Baseline prompt summary:

```text
Classify each T-Mobile Reddit post/comment into exactly one of:
help_request
customer_complaint
employee_complaint
deal_or_policy_info

Respond with only the label name.
```

The baseline produced parseable labels for all test examples:

```text
35 / 35 parseable responses
```

---

## Evaluation Results

The test set contained:

```text
35 examples
```

Overall results:

| Model                   | Accuracy |
| ----------------------- | -------: |
| Zero-shot Groq baseline |    0.771 |
| Fine-tuned DistilBERT   |    0.714 |

Fine-tuning result:

```text
Fine-tuning regression: 0.057
```

The fine-tuned DistilBERT model performed 5.7 percentage points worse than the Groq zero-shot baseline. This suggests that fine-tuning did not outperform the larger LLM on this small and imbalanced dataset.

---

## Baseline Per-Class Metrics

| Label                 | Precision | Recall | F1-score | Support |
| --------------------- | --------: | -----: | -------: | ------: |
| `help_request`        |      1.00 |   0.76 |     0.87 |      17 |
| `customer_complaint`  |      0.20 |   0.50 |     0.29 |       2 |
| `employee_complaint`  |      0.50 |   1.00 |     0.67 |       2 |
| `deal_or_policy_info` |      0.85 |   0.79 |     0.81 |      14 |
| **Accuracy**          |           |        | **0.77** |      35 |
| **Macro avg**         |      0.64 |   0.76 |     0.66 |      35 |
| **Weighted avg**      |      0.86 |   0.77 |     0.80 |      35 |

The Groq baseline performed well on `help_request` and `deal_or_policy_info`, but struggled with `customer_complaint`. One reason is that the test set only contained two `customer_complaint` examples, so one or two mistakes had a large effect on the score.

---

## Fine-Tuned Model Per-Class Metrics

| Label | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `help_request` | 0.76 | 0.76 | 0.76 | 17 |
| `customer_complaint` | 0.00 | 0.00 | 0.00 | 2 |
| `employee_complaint` | 0.00 | 0.00 | 0.00 | 2 |
| `deal_or_policy_info` | 0.67 | 0.86 | 0.75 | 14 |
| **Accuracy** |  |  | **0.714** | 35 |
| **Macro avg** | 0.36 | 0.41 | 0.38 | 35 |
| **Weighted avg** | 0.64 | 0.71 | 0.67 | 35 |

The fine-tuned DistilBERT model performed best on `help_request` and `deal_or_policy_info`, which were also the two largest classes in the dataset. It completely missed `customer_complaint` and `employee_complaint` on the test set, with 0.00 precision, recall, and F1-score for both labels.

This suggests that the fine-tuned model learned the majority-class patterns better than the minority classes. Since there were only 2 `customer_complaint` and 2 `employee_complaint` examples in the test set, the model had very little opportunity to show reliable performance on those categories. The model likely over-predicted the larger labels instead of learning the smaller label boundaries well.

## Confusion Matrix

The notebook generated a confusion matrix image:

```text
outputs/confusion_matrix.png
```

Markdown confusion matrix table:

Replace the TODO values below with the exact values from the confusion matrix image.

| True Label ↓ / Predicted Label → | `help_request` | `customer_complaint` | `employee_complaint` | `deal_or_policy_info` |
| -------------------------------- | -------------: | -------------------: | -------------------: | --------------------: |
| `help_request`                   |             13 |                    0 |                    0 |                      4|
| `customer_complaint`             |              1 |                    0 |                    0 |                      1|
| `employee_complaint`             |              1 |                    0 |                    0 |                      1|
| `deal_or_policy_info`            |              2 |                    0 |                    0 |                     12|

The confusion matrix shows that the fine-tuned model mostly predicted only two labels: help_request and deal_or_policy_info. It correctly classified 13 out of 17 help_request examples and 12 out of 14 deal_or_policy_info examples. However, it did not correctly classify any customer_complaint or employee_complaint examples.
This explains why customer_complaint and employee_complaint had 0.00 precision, recall, and F1-score. The model never predicted those two labels on the test set. This likely happened because the dataset was imbalanced: customer_complaint and employee_complaint had far fewer examples than the majority labels.
The main failure pattern was that the model collapsed the smaller labels into the larger labels. Customer complaints and employee complaints were predicted as either help_request or deal_or_policy_info. This suggests the model learned surface-level signals from the larger classes, such as question wording or policy-related terms, but did not learn enough examples of complaint language to separate those categories reliably.


---

## Error Analysis

The fine-tuned model made 10 wrong predictions out of 35 test examples.

### Wrong Prediction 1

Text excerpt:

> nice thats great. Super helpful...

True label:

```text
help_request
```

Predicted label:

```text
deal_or_policy_info
```

Analysis:

The model likely focused on T-Mobile-related informational wording but missed that the comment was part of a help-seeking exchange. This shows that the model sometimes struggles when the post is short and lacks enough context.

---

### Wrong Prediction 2

Text excerpt:

> It is pretty embarrassing. The company...

True label:

```text
customer_complaint
```

Predicted label:

```text
help_request
```

Analysis:

This was a complaint, but the model predicted `help_request`. This suggests the model may over-predict `help_request`, especially when it sees language that sounds like a problem or issue. The model did not fully distinguish between someone asking for help and someone expressing dissatisfaction.

---

### Wrong Prediction 3

Text excerpt:

> The pros: schedule flexibility...

True label:

```text
employee_complaint
```

Predicted label:

```text
help_request
```

Analysis:

This example came from an employee/workplace context. The model missed the worker perspective and treated the post more like a general discussion or request. This likely happened because `employee_complaint` was underrepresented in the dataset, with only 10 total examples.

---

### Error Pattern Summary

The main failure pattern was that the fine-tuned model confused labels when posts contained mixed signals. A post could include a question mark, policy words, and complaint language all at once. The model often focused on surface keywords instead of the main purpose of the post.

The hardest label boundary was:

```text
help_request vs. deal_or_policy_info
```

A second difficult boundary was:

```text
help_request vs. customer_complaint
```

This shows that the model learned some useful patterns, but it did not always capture the deeper intent of the post.

---

## Sample Classifications

The table below shows examples from the fine-tuned DistilBERT model output. Some examples were correct, while others show the model's main failure pattern: confusing smaller classes with the larger `help_request` and `deal_or_policy_info` labels.

| Text Excerpt | Predicted Label | Confidence | Correct? | Notes |
|---|---|---:|---|---|
| A post asking for help with a T-Mobile issue was classified as `help_request`. | `help_request` | N/A | Yes | This prediction is reasonable because the main purpose of the text was asking for advice or troubleshooting. |
| A post explaining a T-Mobile promotion, bill credit, or eligibility rule was classified as `deal_or_policy_info`. | `deal_or_policy_info` | N/A | Yes | This prediction is reasonable because the post was mainly informational rather than emotional or workplace-focused. |
| "It is pretty embarrassing. The company..." | `help_request` | 0.31 | No | The true label was `customer_complaint`. The model treated a complaint about the company as if the user was asking for help. |
| "The pros: schedule flexibility..." | `help_request` | 0.31 | No | The true label was `employee_complaint`. The model missed the employee/workplace perspective and predicted the larger `help_request` class. |
| "Yea your most likely looking at Optic..." | `help_request` | 0.31 | No | The true label was `deal_or_policy_info`. The model saw explanatory language but classified it as a help request, showing confusion between advice and policy information. |

Because the notebook's wrong-prediction output printed confidence values for incorrect examples, those confidence values are included where available. For the correct examples, I summarized representative correct predictions from the model behavior rather than quoting full Reddit text. The important pattern is that the model performed better on the majority labels and struggled with the smaller complaint-based labels.

## Reflection: What the Model Learned vs. What I Intended

I intended the model to learn the main purpose of each T-Mobile Reddit post or comment. The goal was not just to recognize keywords, but to understand whether the writer was asking for help, complaining as a customer, complaining as an employee, or sharing deal/policy information.

The model learned some of these patterns, especially for `help_request` and `deal_or_policy_info`, which had the most examples in the dataset. However, it struggled with smaller classes like `customer_complaint` and `employee_complaint`. This suggests the model learned the distribution of the dataset as much as the label definitions.

The biggest gap was that the model sometimes treated any problem-related post as a help request, even when the writer was mainly complaining. It also confused policy terms with deal/policy information even when the post was actually asking for help. This means the model relied heavily on surface keywords instead of consistently identifying the writer's main intent.

If I improved this project, I would collect more examples for `customer_complaint` and `employee_complaint`, especially difficult examples that contain mixed signals. I would also manually balance the dataset better before training.

---

## Spec Reflection

One way the spec helped was by forcing me to define the labels before collecting and training the model. Writing the label definitions and edge-case rules made it easier to decide how to label ambiguous posts.

One way my implementation diverged from the original plan was the data collection method. I originally wanted to use Reddit API/PRAW, but Reddit API access was not available in the simple self-service way I expected. Because of that, I used an Reddit scraper to collect public Reddit posts and comments instead. This changed the workflow, but it still produced a usable public dataset for the classifier.

Another divergence was dataset balance. I planned to collect a more even number of examples for each label, but the final dataset had far more `help_request` and `deal_or_policy_info` examples than `customer_complaint` and `employee_complaint`. This imbalance became one of the likely reasons the fine-tuned model struggled on the smaller classes.

---

## AI Usage

### AI Use 1: Label taxonomy and planning

I used ChatGPT to help brainstorm and refine the label taxonomy for T-Mobile Reddit discussions. I directed the AI to help separate customer help requests, customer complaints, employee complaints, and deal/policy information. I revised the labels by adding `employee_complaint` because employee-focused posts are common in T-Mobile-related discussions and represent a different perspective from customer complaints.

What I reviewed or changed:

* I checked that the labels were mutually exclusive enough for the project.
* I added decision rules for hard cases, such as frustrated help requests.
* I kept the labels limited to four because the project required 2-4 labels.

---

### AI Use 2: Dataset preparation script

I used Claude to help write `prepare_dataset.py`, a script that reads the raw CSV, cleans the text, removes unusable rows, applies first-pass labels, and writes `data/takemeter_dataset.csv`.

What I reviewed or changed:

* I changed the input path so the script worked in my local VS Code project instead of Claude's environment.
* I checked that the output CSV had exactly the required columns: `text`, `label`, and `notes`.
* I verified that the labels matched the notebook's `LABEL_MAP`.
* I checked the label distribution to make sure no single label was more than 70% of the dataset.

---

### AI Use 3: Failure analysis support

I used ChatGPT to help interpret the evaluation results after the notebook printed wrong predictions. I asked it to identify patterns in the model's mistakes, then compared that explanation to the actual wrong predictions and the model scores.

What I reviewed or changed:

* I verified that the main pattern was confusion between `help_request` and `deal_or_policy_info`.
* I connected the mistakes to the dataset imbalance and short/mixed-context posts.
* I did not treat the AI analysis as final until I checked it against the notebook outputs.

---

## Files Included

```text
planning.md
README.md
data/takemeter_dataset.csv
prepare_dataset.py
outputs/evaluation_results.json
outputs/confusion_matrix.png
requirements.txt
```

---

## How to Reproduce

1. Open the TakeMeter starter notebook in Google Colab.
2. Set runtime to T4 GPU.
3. Upload `data/takemeter_dataset.csv`.
4. Use this label map:

```python
LABEL_MAP = {
    "help_request": 0,
    "customer_complaint": 1,
    "employee_complaint": 2,
    "deal_or_policy_info": 3,
}
```

5. Run Sections 1 and 2 to load and split the dataset.
6. Run Section 5 to collect the Groq baseline.
7. Run Sections 3 and 4 to fine-tune and evaluate DistilBERT.
8. Run Section 6 to export `evaluation_results.json` and `confusion_matrix.png`.

---

## Demo Video Plan

The demo video shows:

1. The labeled dataset CSV with the four labels.
2. The Colab notebook classifying test examples.
3. Three to five posts being classified with label and confidence visible.
4. One correct prediction and why it makes sense.
5. One incorrect prediction and why it failed.
6. The evaluation report, including baseline accuracy, fine-tuned accuracy, and the confusion matrix.
