# TakeMeter Planning

## Project Overview

TakeMeter is a text classification project that evaluates the type of discourse found in public T-Mobile-related Reddit discussions. The classifier will read a post or comment and assign it to one of four labels: `help_request`, `customer_complaint`, `employee_complaint`, or `deal_or_policy_info`.

The goal is not just to classify text correctly, but to understand what the model learns from messy real-world online discussion. I will design the label taxonomy, collect and label a dataset, fine-tune a DistilBERT classifier, compare it to a zero-shot Groq baseline, and analyze the model's mistakes.

## Community

I chose public T-Mobile Reddit discussions as my online community. This community is a good fit for a classification task because the discourse is active, text-heavy, and varied. Posts often include customers asking for help, customers complaining about billing or service problems, employees talking about workplace frustrations, and users sharing plan, promotion, or policy information.

These distinctions matter because readers respond differently depending on the type of post. A help request needs troubleshooting, a customer complaint shows frustration with the company or service, an employee complaint reveals workplace experience, and a deal or policy post is mainly informational. Because these categories appear often in T-Mobile-related discussions, they are useful labels for a classifier.

## Label Taxonomy

### Label 1: `help_request`

A post is labeled `help_request` if the writer is asking for advice, troubleshooting, or clarification about a T-Mobile issue, such as billing, service, plans, devices, upgrades, promotions, or account problems.

Clear example 1:

> My bill went up after changing plans. Can someone explain what might have happened?

Clear example 2:

> I switched to eSIM and now my service keeps dropping. Has anyone fixed this before?

Boundary notes:

A post can sound frustrated and still be a `help_request` if the main purpose is asking how to solve a problem. The question or request for advice is the strongest signal.

---

### Label 2: `customer_complaint`

A post is labeled `customer_complaint` if the writer is mainly complaining from a customer perspective about T-Mobile service, billing, coverage, support, store experience, promotions, or account problems.

Clear example 1:

> T-Mobile support keeps transferring me and nobody can fix my account issue.

Clear example 2:

> My coverage has gotten worse in my area and I am tired of paying this much for unreliable service.

Boundary notes:

If the post is mostly venting or criticizing a customer experience, it should be labeled `customer_complaint`. If it asks a clear troubleshooting question, it may be `help_request` instead.

---

### Label 3: `employee_complaint`

A post is labeled `employee_complaint` if the writer is mainly complaining from an employee or worker perspective about store management, quotas, commission, scheduling, leadership, customers, job stress, or workplace expectations.

Clear example 1:

> The sales goals keep going up, but staffing keeps getting worse. It feels impossible to hit numbers and still help customers properly.

Clear example 2:

> Management keeps pushing add-ons even when customers clearly do not need them, and employees are the ones who get blamed when people complain.

Boundary notes:

The key difference between `employee_complaint` and `customer_complaint` is point of view. If the writer is describing the issue as an employee, worker, rep, store employee, or salesperson, it belongs in `employee_complaint`.

---

### Label 4: `deal_or_policy_info`

A post is labeled `deal_or_policy_info` if it mainly shares or explains information about T-Mobile promotions, plans, upgrades, trade-ins, pricing, policy changes, account rules, or eligibility.

Clear example 1:

> The new trade-in promo gives higher credits on Go5G Plus than on older plans.

Clear example 2:

> If you cancel a line too soon after adding a promotion, you may lose the monthly bill credits.

Boundary notes:

This label is for informational posts. If the post mainly explains a rule, promotion, plan change, or account policy, it should be labeled `deal_or_policy_info`, even if users may discuss the information positively or negatively in replies.

## Hard Edge Cases

### Edge Case 1: `help_request` vs. `customer_complaint`

Example:

> My bill is wrong again and support has been useless. Does anyone know how to get this fixed?

Possible labels:

* `help_request`
* `customer_complaint`

Decision rule:

If the main goal is asking how to fix the issue, label it `help_request`, even if the writer sounds frustrated. If the main goal is venting or criticizing T-Mobile without asking for a solution, label it `customer_complaint`.

Decision for this example:

`help_request`, because the writer asks how to get the issue fixed.

---

### Edge Case 2: `customer_complaint` vs. `deal_or_policy_info`

Example:

> T-Mobile changed the autopay discount rules and now credit cards do not qualify anymore. This is frustrating.

Possible labels:

* `customer_complaint`
* `deal_or_policy_info`

Decision rule:

If the post mainly explains what changed, label it `deal_or_policy_info`. If the post mainly focuses on anger, frustration, or criticism, label it `customer_complaint`.

Decision for this example:

`deal_or_policy_info`, because the main content is explaining a policy change. The frustration is secondary.

---

### Edge Case 3: `employee_complaint` vs. `deal_or_policy_info`

Example:

> At my store, managers keep telling us to push certain add-ons because the commission structure rewards it more than basic upgrades.

Possible labels:

* `employee_complaint`
* `deal_or_policy_info`

Decision rule:

If the writer is mainly explaining an employee policy or commission structure in a neutral informational way, label it `deal_or_policy_info`. If the writer is complaining about workplace pressure, management, quotas, or job expectations, label it `employee_complaint`.

Decision for this example:

`employee_complaint`, because the post is about pressure from managers and workplace expectations.

## Data Collection Plan

## Data Collection Plan

I will collect public T-Mobile-related Reddit posts and comments from r/tmobile and related public T-Mobile discussion search results. My original plan was to use Reddit API access through PRAW, but Reddit API self-service access was not available during my project timeline. Because of that, I used an Reddit scraper to collect public Reddit data.

The raw export will be saved as:

data/raw_reddit.csv

The final labeled dataset will be saved as:

data/takemeter_dataset.csv

The final dataset will have exactly these columns:

text,label,notes

I will only keep public post/comment text needed for classification. I will remove or avoid usernames, profile links, account numbers, phone numbers, screenshots with private information, and anything from internal T-Mobile systems.

My target is at least 200 labeled examples across these four labels:

- help_request
- customer_complaint
- employee_complaint
- deal_or_policy_info

If one label is underrepresented after scraping, I will search for more examples using targeted search terms. For example, I will search terms like "bill question" and "upgrade question" for help requests, "bad coverage" and "support issue" for customer complaints, "commission" and "sales goals" for employee complaints, and "trade in promo" and "bill credits" for deal or policy information.

I will use a preparation script to clean the raw CSV, remove deleted/removed posts, remove very short examples, remove duplicates, and create the final CSV. I may use rule-based or AI-assisted pre-labeling to speed up annotation, but I will review the labels before using the dataset for training.

## Annotation Process

I will label each example manually using the definitions in this planning document. For each post/comment, I will ask:

1. Is the writer asking for help or troubleshooting?
2. Is the writer complaining as a customer?
3. Is the writer complaining as an employee or worker?
4. Is the writer mainly sharing deal, plan, promotion, or policy information?

If a post could fit more than one label, I will use the edge-case decision rules above. I will write notes for difficult examples, especially examples that sit between two labels. I will include at least three difficult-to-label examples and my decisions in the README.

Because I used a scraper, I expect some rows to be noisy, duplicated, too short, deleted, or unclear. I will skip rows that do not contain enough context to label confidently.

I used a rule-based preparation script as a first labeling pass. The script looked for signals such as question wording for `help_request`, frustration/customer-service language for `customer_complaint`, workplace terms like quota, commission, manager, and store for `employee_complaint`, and promotion/policy terms like bill credit, trade-in, plan, and eligibility for `deal_or_policy_info`.

I will treat these labels as pre-labels, not final unquestioned truth. I will review the dataset for obvious mistakes and document difficult examples in the README.

I may use an AI tool to suggest preliminary labels for small batches, but I will personally review and correct every label before including it in the dataset. If I use AI for pre-labeling, I will disclose that in the README AI usage section.

## Model and Training Plan

The fine-tuned model will use `distilbert-base-uncased` in the provided TakeMeter Colab notebook. The notebook will split the dataset into train, validation, and test sets using a 70% / 15% / 15% split.
 
Initial hyperparameters:

* Base model: `distilbert-base-uncased`
* Training platform: Google Colab with T4 GPU
* Epochs: 3
* Learning rate: 2e-5
* Batch size: 16
* Max token length: 256

I will start with the default hyperparameters because the dataset is small, around 200 examples. Three epochs should give the model enough time to learn the label patterns without overfitting too much. If training results look suspicious, such as very high accuracy or very poor validation performance, I will check for data leakage, label imbalance, or inconsistent labels before changing hyperparameters.

## Baseline Plan

I will compare the fine-tuned DistilBERT model to a zero-shot Groq baseline using `llama-3.3-70b-versatile`. The baseline prompt will include the same label definitions from this planning document and will instruct the model to output only one exact label name.

The baseline is important because it shows whether fine-tuning actually helped. If the zero-shot model performs as well as or better than the fine-tuned model, that may mean my dataset is too small, the labels are too easy for a general LLM, or my annotations are not consistent enough for DistilBERT to learn.

## Evaluation Metrics

I will evaluate both the baseline and fine-tuned model on the same test set.

Metrics I will report:

* Overall accuracy
* Per-class precision, recall, and F1-score
* Confusion matrix
* At least 3 wrong predictions with analysis

Accuracy is useful because it shows the percentage of total examples classified correctly. However, accuracy alone is not enough because the model could perform well overall while failing on one label. For this task, per-class F1-score is important because I need to know whether the model can distinguish all four categories, especially `employee_complaint`, which may be more subtle or less common.

The confusion matrix will help me see which labels the model confuses. For example, if the model often predicts `customer_complaint` when the correct label is `help_request`, that would show the model is over-weighting frustrated language and missing the actual request for help.

## Definition of Success

A good enough result for this project would be:

* Fine-tuned model overall accuracy of at least 70%
* Fine-tuned model improves over the zero-shot Groq baseline, or performs close to it with clearer local/deployable behavior
* Each label has an F1-score of at least 0.60
* No label has an F1-score near 0
* The confusion matrix shows understandable mistakes rather than random guessing

For a real community tool, I would want stronger performance before deployment, such as at least 80% accuracy and at least 0.70 F1-score for every label. For this class project, I will consider the classifier successful if it learns meaningful differences between the labels and the error analysis reveals specific, explainable failure patterns.

## Anticipated Challenges

### Challenge 1: Frustrated help requests

Many T-Mobile posts may include frustration and a question in the same text. This may confuse the model between `help_request` and `customer_complaint`. I will handle this by labeling based on the main purpose of the post. If the writer is asking what to do next, it is `help_request`.

### Challenge 2: Employee vs. customer perspective

Some posts may describe store behavior or sales pressure without clearly saying whether the writer is an employee or customer. This may make `employee_complaint` hard to separate from `customer_complaint`. I will only use `employee_complaint` when the writer clearly speaks from a worker perspective, such as mentioning “my store,” “customers,” “management,” “quotas,” “commission,” or “working here.”

### Challenge 3: Label imbalance

There may be many more customer complaints than employee complaints or policy posts. If the dataset becomes imbalanced, the model may learn to over-predict the majority class. I will monitor label counts while collecting data and intentionally collect more examples for underrepresented labels.

### Challenge 4: Short or low-context posts

Some Reddit comments may be very short, such as “same thing happened to me” or “this company is terrible.” These may not give enough context for a reliable label. If a post does not contain enough information to assign one of the four labels confidently, I will skip it instead of forcing a bad label.

### Challenge 5: dataset imbalance after scraping

Even though I planned for a balanced dataset, scraping public Reddit data may produce more help requests and deal/policy posts than customer or employee complaints. This can cause the fine-tuned model to perform better on majority classes and worse on underrepresented labels. I will report the final label distribution and discuss any imbalance in my README and error analysis.

## AI Tool Plan

### Label stress-testing

### Annotation assistance

I used Claude to help create a dataset preparation script named `prepare_dataset.py`. I gave Claude my label definitions, decision rules, and raw CSV structure. The script reads the raw CSV, combines title/body fields into one text field, cleans scraper artifacts, removes deleted or very short rows, removes duplicates, applies first-pass rule-based labels, and writes `data/takemeter_dataset.csv`.

I reviewed and changed the script before using it. One important change was fixing the input file path so it worked in my local VS Code project instead of Claude's environment. I also checked that the output CSV had exactly the required columns: `text`, `label`, and `notes`.

I will disclose this AI-assisted annotation workflow in my README because the project allows LLM or AI pre-labeling only if I review and correct the labels myself.

What I will verify:

* Whether each generated example can be assigned to exactly one label
* Whether the edge-case rules are clear enough
* Whether any two labels overlap too much

### Annotation assistance

I may use an LLM to pre-label small batches of examples, but only as a first pass. I will personally review every label before adding it to the final dataset. If the AI suggests a label that does not match my definitions, I will override it. I will track this in the `notes` column when a label required a difficult decision.

What I will verify:

* The AI output uses only the four valid labels
* The suggested label matches my written decision rules
* No example is accepted without human review

### Failure analysis

After training, I will give ChatGPT or Claude a list of wrong predictions from the fine-tuned model and ask it to look for patterns. For example, I will ask whether the model is confusing short posts, sarcastic posts, employee/customer perspective, or help requests written in an angry tone.

What I will verify:

* I will re-read the wrong predictions myself
* I will compare the AI's suggested pattern to the confusion matrix
* I will only include patterns in the README if they are supported by the actual examples

## Planned Files

Repository name:

```text
ai201-project3-takemeter
```

Planned structure:

```text
ai201-project3-takemeter/
├── planning.md
├── README.md
├── data/
│   └── takemeter_dataset.csv
└── outputs/
    ├── evaluation_results.json
    └── confusion_matrix.png
```

The Colab notebook will be saved separately in Google Drive, but the dataset, planning document, README, and output files will be committed to GitHub.

## Dataset CSV Format

The dataset will use this format:

```csv
text,label,notes
"My bill went up after changing plans. Can someone explain what might have happened?",help_request,"asks for explanation"
"T-Mobile support keeps transferring me and nobody can fix my account issue.",customer_complaint,"main purpose is complaint"
"The sales goals keep going up, but staffing keeps getting worse.",employee_complaint,"employee perspective"
"If you cancel a line too soon after adding a promotion, you may lose the monthly bill credits.",deal_or_policy_info,"policy explanation"
```

## Notebook Label Map

When I run the Colab notebook, I will use this label map:

```python
LABEL_MAP = {
    "help_request": 0,
    "customer_complaint": 1,
    "employee_complaint": 2,
    "deal_or_policy_info": 3,
}
```

## Baseline Prompt Draft

```text
You are classifying public Reddit posts and comments from T-Mobile-related discussion communities.
Assign each post to exactly one of the following categories.

help_request: The post is asking for advice, troubleshooting, or clarification about a T-Mobile issue, such as billing, service, plans, devices, upgrades, or account problems.
Example: "My bill went up after changing plans. Can someone explain what might have happened?"

customer_complaint: The post is mainly complaining from a customer perspective about T-Mobile service, billing, coverage, support, store experience, promotions, or account problems.
Example: "T-Mobile support keeps transferring me and nobody can fix my account issue."

employee_complaint: The post is mainly complaining from an employee or worker perspective about store management, quotas, commission, scheduling, leadership, customers, job stress, or workplace expectations.
Example: "The sales goals keep going up, but staffing keeps getting worse, and employees are the ones getting blamed."

deal_or_policy_info: The post is mainly sharing or explaining information about T-Mobile promotions, plans, upgrades, trade-ins, pricing, policy changes, account rules, or eligibility.
Example: "If you cancel a line too soon after adding a promotion, you may lose the monthly bill credits."

Decision rules:
- If the writer is asking how to fix something, choose help_request even if they sound frustrated.
- If the writer is complaining as a customer, choose customer_complaint.
- If the writer is complaining as a worker or employee, choose employee_complaint.
- If the post mainly explains a plan, promo, trade-in, price, rule, or eligibility detail, choose deal_or_policy_info.

Respond with ONLY the label name.
Do not explain your reasoning.

Valid labels:
help_request
customer_complaint
employee_complaint
deal_or_policy_info
```

## Stretch Features

I am not planning to start a stretch feature yet. If I decide to add one later, I will update this planning document before starting it.
